#%% MODULES
# %reset -sf
from herp.derp.exercises import verificar_aprendizaje, evolucion_version_alterada, verificar_estados_espureos, minimo_local_desde_original
#%% 1a) VERIFICACION DE APRENDIZAJE

verificar_aprendizaje('images.txt')

#%% 1b) VERSIONES ALTERADAS

evolucion_version_alterada('images_ejb.txt')

#%% 1c) ESTADOS ESPUREOS

verificar_estados_espureos('images_ejc_1.txt')
# verificar_estados_espureos('images_ejc_2.txt')
# verificar_estados_espureos('images_ejc_3.txt')

#%%% 1d) MINIMO LOCAL DESDE ORIGINAL

minimo_local_desde_original('images.txt')

#TODO:
    # Entender porque hay combinaciones lineales que no son estados espureos.
    # Unificar el uso de los vectores con imagenes
        # A veces los necesito en fila y otras veces en columna, pero ya perdí
        #la noción de cuando los devuelvo en fila y cuando en columna.