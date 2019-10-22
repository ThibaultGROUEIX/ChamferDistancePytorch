def get(dim=3):
    if dim == 2:
        import chamfer_2D.dist_chamfer_idx as ext
        distChamfer = ext.chamferDist()
    elif dim == 3:
        import chamfer_3D.dist_chamfer_idx as ext
        distChamfer = ext.distChamfer
    elif dim == 5:
        import chamfer_3D.dist_chamfer_idx as ext
        distChamfer = ext.distChamfer
    else:
        print('missing dim')
    return distChamfer