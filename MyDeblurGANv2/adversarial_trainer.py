import torch
import copy


class GANFactory:
    factories = {}

    def __init__(self):
        pass

    def add_factory(gan_id, model_factory):
        GANFactory.factories.put[gan_id] = model_factory

    add_factory = staticmethod(add_factory)

    # A Template Method:

    def create_model(gan_id, net_d=None, criterion=None):
        if gan_id not in GANFactory.factories:
            GANFactory.factories[gan_id] = \
                eval(gan_id + '.Factory()')
        return GANFactory.factories[gan_id].create(net_d, criterion)

    create_model = staticmethod(create_model)


class GANTrainer(object):
    def __init__(self, net_d, criterion):
        self.net_d = net_d
        self.criterion = criterion

    def loss_d(self, pred, gt):
        pass

    def loss_g(self, pred, gt):
        pass

    def get_params(self):
        pass




class SingleGAN(GANTrainer):
    def __init__(self, net_d, criterion):
        GANTrainer.__init__(self, net_d, criterion)
        self.net_d = self.net_d.cuda()

    def loss_d(self, pred, gt):
        return self.criterion(self.net_d, pred, gt)

    def loss_g(self, pred, gt):
        return self.criterion.get_g_loss(self.net_d, pred, gt)

    def get_params(self):
        return self.net_d.parameters()

    class Factory:
        @staticmethod
        def create(net_d, criterion): return SingleGAN(net_d, criterion)




