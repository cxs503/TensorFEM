"""Render actual displacement and von-Mises fields from hull-girder JSON."""
import argparse, json
from pathlib import Path
def main():
    p=argparse.ArgumentParser(); p.add_argument("report",type=Path); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    data=json.loads(a.report.read_text()); a.output_dir.mkdir(parents=True,exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        raster=True
    except ImportError:
        raster=False
    def svg(points, value, title, path, color="#d73027"):
        xs=[q["x"] for q in points]; ys=[q["y"] for q in points]; vs=[q[value] for q in points]
        lo,hi=min(vs),max(vs); span=hi-lo or 1.0
        circles=[]
        for x,y,v in zip(xs,ys,vs):
            r=int(40+180*(v-lo)/span); circles.append(f'<circle cx="{60+860*x/(max(xs) or 1):.1f}" cy="{260-190*y/(max(ys) or 1):.1f}" r="{max(2,r//25)}" fill="{color}" fill-opacity="{0.35+0.6*(v-lo)/span:.3f}"/>')
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="320"><rect width="100%" height="100%" fill="white"/><text x="20" y="25" font-family="sans-serif">{title}</text>'+''.join(circles)+'</svg>')
    for i,c in enumerate(data["cases"]):
        f=c["field"]; nx=c["mesh"]["nx"]; ny=c["mesh"]["ny"]
        for key,label,fn in (("uy","FE transverse displacement [m]","displacement"),):
            pts=f["nodes"]
            if raster:
                fig,ax=plt.subplots(figsize=(8,3)); sc=ax.scatter([q["x"] for q in pts],[q["y"] for q in pts],c=[q[key] for q in pts],s=16,cmap="turbo"); fig.colorbar(sc,ax=ax,label=label); ax.set_title(f"Hull girder Q4 {nx}x{ny}: displacement"); fig.tight_layout(); fig.savefig(a.output_dir/f"hull_girder_{nx}x{ny}_{fn}.png",dpi=160); plt.close(fig)
            else: svg(pts,key,f"Hull girder Q4 {nx}x{ny}: displacement",a.output_dir/f"hull_girder_{nx}x{ny}_{fn}.svg")
        pts=f["element_stress"]
        if raster:
            fig,ax=plt.subplots(figsize=(8,3)); sc=ax.scatter([q["x"] for q in pts],[q["y"] for q in pts],c=[q["von_mises"] for q in pts],s=22,cmap="turbo"); fig.colorbar(sc,ax=ax,label="von Mises stress [Pa]"); ax.set_title(f"Hull girder Q4 {nx}x{ny}: von Mises"); fig.tight_layout(); fig.savefig(a.output_dir/f"hull_girder_{nx}x{ny}_von_mises.png",dpi=160); plt.close(fig)
        else: svg(pts,"von_mises",f"Hull girder Q4 {nx}x{ny}: von Mises",a.output_dir/f"hull_girder_{nx}x{ny}_von_mises.svg")
if __name__ == "__main__": main()
