"""Independent NumPy bilinear force/couple/work reconstruction of actual FVM loads."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fvm',type=Path,default=Path('/home/jsyc/tensor-suite-development/TensorFVM'));args=parser.parse_args()
    folder=ROOT/'docs/assets/fvm-wall-exchange';study=json.loads((folder/'study.json').read_text())
    if study['physical_accuracy_qualified'] is not False:raise ValueError('false wall exchange qualification')
    for name,sha in study['source_sha256'].items():
        repo,rel=name.split('/',1);path=(ROOT if repo=='TensorFEM' else args.fvm)/rel
        if digest(path)!=sha:raise ValueError('wall exchange source hash mismatch')
    rows=[]
    for case,info in study['cases'].items():
        path=folder/(case+'.json');source=args.fvm/'docs/conservative-transport'/case/'wall-facets.json'
        if digest(path)!=info['artifact_sha256'] or digest(source)!=info['source_raw_sha256']:raise ValueError('wall exchange raw hash mismatch')
        d=json.loads(path.read_text());src=json.loads(source.read_text())
        if d['physical_accuracy_qualified'] is not False or d['flexible_response_solved'] is not False or d['fluid_feedback'] is not False:raise ValueError('false physics/feedback claim')
        points=np.asarray(d['force_points_m']);nodes=np.asarray(d['carrier_nodes_m']);v=np.asarray(d['nodal_virtual_velocity'])
        xi=2*(points[:,0]-nodes[0,0])/(nodes[1,0]-nodes[0,0])-1
        eta=2*(points[:,1]-nodes[0,1])/(nodes[3,1]-nodes[0,1])-1
        weights=np.stack(((1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)),1)/4
        anchors=weights@nodes;offsets=points-anchors;returned=weights@v[:,:3]+np.cross(weights@v[:,3:],offsets)
        motion=float(np.max(abs(returned-np.asarray(d['returned_point_velocity_m_s']))))
        source_points=[];source_pressure=[];source_viscous=[]
        for facet in src['facets']:
            source_points.extend(facet['points_m']);source_pressure.extend(facet['pressure_force_on_body_N']);source_viscous.extend(facet['viscous_force_on_body_N'])
        if not np.array_equal(points,source_points) or not np.array_equal(np.asarray(d['pressure_force_N']),source_pressure) or not np.array_equal(np.asarray(d['viscous_force_N']),source_viscous):raise ValueError('actual source points/components changed')
        errors={}
        for label,field in [('pressure','pressure_force_N'),('viscous','viscous_force_N'),('total','total_force_N')]:
            f=np.asarray(d[field]);g=np.concatenate((weights.T@f,weights.T@np.cross(offsets,f)),axis=1)
            error=float(np.max(abs(g-np.asarray(d['generalized_loads'][label]))));force=np.sum(g[:,:3],axis=0);moment=np.sum(np.cross(nodes,g[:,:3])+g[:,3:],axis=0)
            moment_error=float(np.max(abs(moment-np.cross(points,f).sum(0))));force_error=float(np.max(abs(force-f.sum(0))));power_error=abs(float(np.sum(g*v)-np.sum(f*returned)))
            errors[label]={'generalized_force_error':error,'force_error_N':force_error,'moment_error_Nm':moment_error,'power_error_W':power_error}
            if max(error,force_error,moment_error,power_error)>1e-10:raise ValueError('independent wall mapping failed')
        if motion>1e-12 or d['source_time_s']!=src['time_s'] or d['source_viscous_time_s']!=src['viscous_evaluation_time_s']:raise ValueError('wall clock/motion mismatch')
        rows.append({'case':case,'point_count':len(points),'mapping_errors':errors,'returned_motion_error_m_s':motion,'pressure_time_s':d['source_time_s'],'viscous_time_s':d['source_viscous_time_s']})
    report={'schema':'tensorfem.independent-fvm-wall-audit/1','cases':rows,'study_sha256':digest(folder/'study.json'),'auditor_sha256':digest(Path(__file__)),'physical_accuracy_qualified':False,'scope':'NumPy point force/moment/work transfer only; no CFD or flexible response integration'}
    (folder/'audit-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
