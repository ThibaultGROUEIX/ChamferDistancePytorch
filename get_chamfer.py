import sys
def get(dim=3):
    if dim == 2:
        sys.path.insert(0, './extension/chamfer_2D/')
        import dist_chamfer_idx_2D as ext_2D
        distChamfer = ext_2D.chamfer_2DDist()
    elif dim == 3:
        sys.path.insert(0, './extension/chamfer_3D/')
        import dist_chamfer_idx_3D as ext_3D
        distChamfer = ext_3D.chamfer_3DDist()
    elif dim == 5:
        sys.path.insert(0, './extension/chamfer_5D/')
        import dist_chamfer_idx_5D as ext_5D
        distChamfer = ext_5D.chamfer_5DDist()
    else:
        print('missing dim')
    return distChamfer