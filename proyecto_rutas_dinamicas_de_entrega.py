import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium
import random
from scipy.spatial.distance import cdist

# ==========================================
# 1. CONFIGURACIÓN Y MAPA (BOGOTÁ)
# ==========================================
print("Descargando mapa de Chapinero (esto puede tardar unos segundos)...")
# Descargamos un grafo un poco más grande para que haya espacio para todo
G = ox.graph_from_place("Chapinero, Bogotá, Colombia", network_type='drive')

# Centro de operaciones (Depósito)
deposito_nodo = list(G.nodes())[0]
deposito_lat = G.nodes[deposito_nodo]['y']
deposito_lon = G.nodes[deposito_nodo]['x']

# ==========================================
# 2. GENERACIÓN DE DATOS (PEDIDOS E INCIDENTES)
# ==========================================

# A. Generar Pedidos (Tus entregas)
def generar_pedidos(n=20):
    nodos = list(G.nodes())
    seleccionados = random.sample(nodos, n)
    datos = []
    for nodo in seleccionados:
        datos.append([G.nodes[nodo]['y'], G.nodes[nodo]['x'], nodo])
    return pd.DataFrame(datos, columns=['lat', 'lon', 'nodo_id'])

df_pedidos = generar_pedidos(n=20)

# B. Generar Incidentes (Los accidentes/obras)
def generar_incidentes(n=5):
    nodos = list(G.nodes())
    # Nos aseguramos que los incidentes no caigan exactamente sobre un cliente (opcional)
    nodos_disponibles = [x for x in nodos if x not in df_pedidos['nodo_id'].values]
    seleccionados = random.sample(nodos_disponibles, n)
    
    lista_incidentes = []
    tipos = ["Choque", "Obra", "Protesta"]
    for nodo in seleccionados:
        lista_incidentes.append({
            'id': nodo,
            'coords': (G.nodes[nodo]['y'], G.nodes[nodo]['x']),
            'tipo': random.choice(tipos)
        })
    return lista_incidentes

lista_incidentes = generar_incidentes(n=8) # Creamos 8 incidentes en la zona

# ==========================================
# 3. MACHINE LEARNING Y OPTIMIZACIÓN
# ==========================================

# Clustering: Asignar pedidos a vehículos
num_vehiculos = 3
kmeans = KMeans(n_clusters=num_vehiculos, random_state=42).fit(df_pedidos[['lat', 'lon']])
df_pedidos['cluster'] = kmeans.labels_

# Función Solver (OR-Tools)
def resolver_tsp_orden(puntos_coords):
    # Añadimos el depósito al inicio
    ubicaciones = [[deposito_lat, deposito_lon]] + puntos_coords
    
    # Matriz de distancias (Euclidiana para rapidez del ejemplo)
    dist_matrix = cdist(ubicaciones, ubicaciones, metric='euclidean') * 100000
    dist_matrix = dist_matrix.astype(int)
    
    manager = pywrapcp.RoutingIndexManager(len(ubicaciones), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    
    solution = routing.SolveWithParameters(search_parameters)
    
    indices_ordenados = [] # Índices relativos a la lista 'ubicaciones'
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            indices_ordenados.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        indices_ordenados.append(manager.IndexToNode(index)) # Volver al depósito
        
    return indices_ordenados

# ==========================================
# 4. VISUALIZACIÓN INTEGRADA
# ==========================================
m = folium.Map(location=[deposito_lat, deposito_lon], zoom_start=14, tiles='CartoDB positron')

# A. Dibujar el Depósito
folium.Marker(
    [deposito_lat, deposito_lon], popup="<b>CENTRAL</b>", icon=folium.Icon(color='black', icon='home')
).add_to(m)

# B. Dibujar Incidentes (CAPA DE ALERTA)
ids_incidentes = [inc['id'] for inc in lista_incidentes]

for inc in lista_incidentes:
    folium.Marker(
        location=inc['coords'],
        popup=f"⚠️ {inc['tipo']}",
        icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
    ).add_to(m)

colores = ['blue', 'green', 'purple', 'orange']

# C. Procesar cada vehículo
for v_id in range(num_vehiculos):
    cluster_data = df_pedidos[df_pedidos['cluster'] == v_id]
    
    if len(cluster_data) == 0: continue

    # Preparar datos para OR-Tools
    coords_entrega = cluster_data[['lat', 'lon']].values.tolist()
    ids_nodos_entrega = cluster_data['nodo_id'].values.tolist()
    
    # Obtener el ORDEN óptimo de visita (índices)
    orden_indices = resolver_tsp_orden(coords_entrega)
    
    # Reconstruir la ruta nodo a nodo usando las calles reales (NetworkX)
    # La lista 'orden_indices' tiene: 0 (deposito), 1..N (entregas), 0 (deposito)
    
    nodos_ruta_real = [deposito_nodo] # Empezamos en el depósito
    
    # Mapear índice de OR-Tools a ID de Nodo real
    mapa_nodos = [deposito_nodo] + ids_nodos_entrega
    
    ruta_completa_nodos = []
    
    # Trazar camino real entre paradas
    for i in range(len(orden_indices) - 1):
        origen = mapa_nodos[orden_indices[i]]
        destino = mapa_nodos[orden_indices[i+1]]
        
        # Calcular camino más corto en calles entre paradas
        try:
            camino_parcial = nx.shortest_path(G, origen, destino, weight='length')
            ruta_completa_nodos.extend(camino_parcial[:-1]) # Evitar duplicar el nodo de conexión
        except nx.NetworkXNoPath:
            print(f"No se encontró camino entre nodos {origen} y {destino}")

    ruta_completa_nodos.append(deposito_nodo) # Cerrar ciclo
    
    # D. Verificar si esta ruta pasa por un incidente
    alerta_ruta = False
    incidentes_en_esta_ruta = 0
    for nodo_ruta in ruta_completa_nodos:
        if nodo_ruta in ids_incidentes:
            alerta_ruta = True
            incidentes_en_esta_ruta += 1
            
    # E. Dibujar Ruta y Marcadores
    color_ruta = 'red' if alerta_ruta else colores[v_id]
    grosor = 5 if alerta_ruta else 3
    estilo = 'dash' if alerta_ruta else 'solid' # Línea punteada si hay peligro
    
    # Dibujar línea de calle
    ox.plot_route_folium(G, ruta_completa_nodos, route_map=m, color=color_ruta, weight=grosor, opacity=0.7, dash_array='10, 10' if alerta_ruta else None)
    
    # Dibujar los marcadores de entrega
    for idx, row in cluster_data.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6,
            color=colores[v_id],
            fill=True,
            popup=f"Pedido {idx} (Vehículo {v_id})"
        ).add_to(m)

print(f"Generado mapa con {len(df_pedidos)} pedidos y {len(lista_incidentes)} incidentes.")
m.save("proyecto_final_logistica.html")