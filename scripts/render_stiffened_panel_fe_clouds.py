"""Render real FE displacement and von-Mises clouds from benchmark JSON."""
import argparse, json
from pathlib import Path
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from PIL import Image, ImageDraw

def main():
    p=argparse.ArgumentParser(); p.add_argument('report',type=Path); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
    r=json.loads(a.report.read_text()); c=r['cases'][-1]; f=c['field']; a.output_dir.mkdir(parents=True,exist_ok=True)
    x=[q['x'] for q in f['nodes']]; y=[q['y'] for q in f['nodes']]
    for key,name,label in [('ux','displacement','u_x (m)'),('von_mises','von_mises','von Mises (Pa)')]:
        if key=='ux': v=[q['ux'] for q in f['nodes']]
        else: x,y,v=[q['x'] for q in f['element_stress']],[q['y'] for q in f['element_stress']],[q[key] for q in f['element_stress']]
        if plt is not None:
            plt.figure(figsize=(8,3.5)); plt.scatter(x,y,c=v,s=8,cmap='viridis'); plt.colorbar(label=label); plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.title(f'Stiffened panel FE {label}'); plt.tight_layout(); plt.savefig(a.output_dir/f'{name}.png',dpi=180); plt.close()
        else:
            # Dependency-free fallback: a genuine raster field map.
            w,h=1200,520; im=Image.new('RGB',(w,h),'white'); d=ImageDraw.Draw(im); lo,hi=min(v),max(v)
            for xx,yy,vv in zip(x,y,v):
                px=int(50+(xx-min(x))/(max(x)-min(x))*1100); py=int(30+(max(y)-yy)/(max(y)-min(y))*440); q=(vv-lo)/(hi-lo or 1); d.ellipse((px-2,py-2,px+2,py+2),fill=(int(255*q),60,int(255*(1-q))))
            d.text((50,480),f'{label}  min={lo:.4g} max={hi:.4g}',fill='black'); im.save(a.output_dir/f'{name}.png')
if __name__=='__main__': main()
