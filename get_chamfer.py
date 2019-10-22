def get(opt):
    if opt.dim_template == 2:
        import chamfer_2D.dist_chamfer_idx as ext
        distChamfer = ext.chamferDist()
    elif opt.dim_template == 3:
        import chamfer_3D.dist_chamfer_idx as ext
        distChamfer = ext.distChamfer
    elif opt.dim_template == 5:
        import chamfer_3D.dist_chamfer_idx as ext
        distChamfer = ext.distChamfer
    else:
        print('missing opt.dim_template')
    return distChamfer