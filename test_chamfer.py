import torch
import dist_chamfer as ext

distChamfer = ext.chamferDist()
from torch.autograd import Variable


def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    P = rx.t() + ry - 2 * zz
    return P


def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()


def mydistChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return torch.min(P, 2)[0], torch.min(P, 1)[0]


def test_chamfer():
    distChamfer = ext.chamferDist()
    p1 = torch.rand(4, 100, 3).cuda()
    p2 = torch.rand(4, 100, 3).cuda()
    points1 = Variable(p1, requires_grad=True)
    points2 = Variable(p2)
    dist1, dist2, = distChamfer(points1, points2)

    loss = torch.sum(dist1)
    print(loss)
    loss.backward()
    print(points1.grad, points2.grad)

    mydist1, mydist2 = mydistChamfer(points1, points2)
    d1 = (dist1 - mydist1) ** 2
    d2 = (dist2 - mydist2) ** 2
    print(d1, d2)
    assert (
        torch.sum(d1) + torch.sum(d2) < 0.00000001
    ), "chamfer cuda and chamfer normal are not giving the same results"


test_chamfer()
