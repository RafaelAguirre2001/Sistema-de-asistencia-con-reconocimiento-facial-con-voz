class flabianos:
    """ Lista de invitados """

    def __init__(self):
        self.Invitados=['Jesus Rafael','Leonardo Torres']

    def TuSiTuNo(self,EllosSi):        
        if EllosSi in self.Invitados:
            print('Bienvenido {}'.format(EllosSi))
        else:
            print('Lo siento {}, datos erroneos'.format(EllosSi))
