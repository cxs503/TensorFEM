"""Create VTK, JSON and Markdown artifacts for the qualified hemisphere."""
from pathlib import Path
from tensorfem.hemisphere_postprocess import (
    build_hemisphere_post,write_hemisphere_json,write_hemisphere_markdown,
    write_hemisphere_vtk,
)
from tensorfem.spherical_shell import hemisphere_with_hole


if __name__=="__main__":
    out=Path("hemisphere_post");out.mkdir(exist_ok=True)
    post=build_hemisphere_post(hemisphere_with_hole(24))
    write_hemisphere_json(out/"result.json",post)
    write_hemisphere_vtk(out/"result.vtk",post)
    write_hemisphere_markdown(out/"validation.md",post)
    print(post.validation)
