import os
import requests
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime
from calendar import monthrange
from joblib import dump, load
import torch
import torch.nn as nn

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definir modelo PyTorch (misma arquitectura del entrenamiento)
class SequentialLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size1=64, hidden_size2=32, output_size=1, dropout=0.1):
        super(SequentialLSTM, self).__init__()
        
        # Bi-LSTM primera capa
        self.bilstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(hidden_size1 * 2)  # *2 por bidireccional
        self.dropout1 = nn.Dropout(dropout)
        
        # LSTM segunda capa
        self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(dropout / 2)
        
        # Capas densas
        self.dense1 = nn.Linear(hidden_size2, 64)
        self.dropout3 = nn.Dropout(dropout / 2)
        self.dense2 = nn.Linear(64, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Bi-LSTM primera capa
        lstm_out1, _ = self.bilstm1(x)
        
        # BatchNorm requiere (batch, features, seq) -> transponer
        lstm_out1 = lstm_out1.transpose(1, 2)
        lstm_out1 = self.bn1(lstm_out1)
        lstm_out1 = lstm_out1.transpose(1, 2)
        lstm_out1 = self.dropout1(lstm_out1)
        
        # LSTM segunda capa
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # BatchNorm
        lstm_out2 = lstm_out2.transpose(1, 2)
        lstm_out2 = self.bn2(lstm_out2)
        lstm_out2 = lstm_out2.transpose(1, 2)
        lstm_out2 = self.dropout2(lstm_out2)
        
        # Aplicar capas densas a cada timestep
        output = self.dense1(lstm_out2)
        output = self.relu(output)
        output = self.dropout3(output)
        output = self.dense2(output)
        
        return output

class ProyeccionGrupo:
    def __init__(self):
        # Mapeo de grupos de marcas actualizadas para PyTorch
        self.grupos_marcas = [
            {'nombre': 'TOYOTA', 'scaler': 'toyota', 'bdproy': 'Toyota'},
            {'nombre': 'OM', 'scaler': 'om', 'bdproy': 'OM'},
            {'nombre': 'HINO', 'scaler': 'hino', 'bdproy': 'Hino'}
        ]
        
        # Obtener fecha actual
        self.current_year, self.current_month = datetime.now().year, datetime.now().month
        self.current_per = datetime.now().strftime("%Y%m")
        self.days_month = monthrange(self.current_year, self.current_month)[1]
        self.current_day = datetime.now().day

    def ejecutar_proyecciones(self):
        """Método principal para ejecutar proyecciones para todos los grupos"""
        print(f"Usando dispositivo: {device}")
        print(f"Fecha actual: {self.current_day}/{self.current_month}/{self.current_year}")
        print(f"Días en el mes: {self.days_month}")
        
        for grupo in self.grupos_marcas:
            try:
                self.procesar_grupo(grupo['nombre'], grupo['scaler'], grupo['bdproy'])
            except Exception as e:
                print(f"Error al procesar {grupo['nombre']}: {e}")
                
        print("Proceso completado para todos los grupos de marcas.")

    def procesar_grupo(self, grupo_marca, scaler_grupo, bdproy_grupo):
        """Procesa un grupo de marca específico y gestiona la interacción con el usuario"""
        print(f"\n{'='*50}")
        print(f"Procesando grupo: {grupo_marca}")
        print(f"{'='*50}")
        
        # Obtener datos de proyección matriz
        df = self.obtener_datos_proyeccion(grupo_marca)
        if df is None:
            return
        
        # Filtrar datos para activaciones y dominicales del periodo actual
        df_act_p = df[(df['periodo'] == self.current_per) & (df['tipo'] == 'acts')]
        df_dom_p = df[(df['periodo'] == self.current_per) & (df['tipo'] == 'doms')]
        
        if df_act_p.empty or df_dom_p.empty:
            print(f"No hay datos suficientes para {grupo_marca} en el periodo actual")
            return
        
        # Preparar arrays para predicción
        pred_acts = np.array(df_act_p.iloc[:, 1:])  # Excluir columna 'periodo'
        pred_doms = np.array(df_dom_p.iloc[:, 1:])  # Excluir columna 'periodo'
        
        # Asegurar 31 días
        pred_acts = self.ensure_31_days(pred_acts)
        pred_doms = self.ensure_31_days(pred_doms)
        
        # Convertir a float32 y procesar dominicales
        pred_acts = pred_acts.astype(np.float32)
        pred_doms = pred_doms.astype(np.float32)
        pred_doms = np.nan_to_num(pred_doms, nan=0.0)
        pred_doms = (pred_doms >= 0.5).astype(np.float32)
        
        # Cargar el scaler y normalizar datos
        scaler = self.cargar_scaler(scaler_grupo)
        if scaler is None:
            return
        
        pred_acts_norm = scaler.transform(pred_acts)
        
        # Calcular días para predicción (N = días conocidos hasta ahora)
        N = self.current_day + (31 - self.days_month) - 1  # Ajuste para índice 0-based
        if N < 0:
            N = 0
        elif N >= 31:
            N = 30
            
        print(f"Días conocidos (N): {N+1}, Días a predecir: {31-N-1}")
        
        # Cargar modelo PyTorch
        model = self.cargar_modelo_pytorch(scaler_grupo, N)
        if model is None:
            return
        
        # Preparar datos para predicción
        # Acts parciales: conocidos hasta N, desconocidos puestos a 0
        acts_partial = pred_acts_norm[0].copy()
        acts_partial[N+1:] = 0.0
        
        # Entrada multicanal (1, 31, 2)
        X = np.stack([acts_partial, pred_doms[0]], axis=-1)[np.newaxis, ...]
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Realizar predicción
        model.eval()
        with torch.no_grad():
            predicciones_norm = model(X_tensor).cpu().numpy()[0, :, 0]  # (31,)
        
        print(f"Predicciones (datos normalizados) para {grupo_marca}:")
        print(f"Últimos 10 valores: {predicciones_norm[-10:]}")
        
        # Transformar de vuelta a escala original
        pred_complete = scaler.inverse_transform([predicciones_norm])
        pred_complete = np.where(pred_complete < 0, 0, pred_complete)  # Eliminar valores negativos
        
        # Aplicar modificador dominical (reducir en días dominicales futuros)
        mod = np.zeros_like(pred_doms[0])
        indices = np.arange(len(pred_doms[0]))
        mask = indices > N + 1  # Días futuros
        mod[mask] = pred_doms[0][mask] * 0.9  # Reducir 90% en dominicales futuros
        
        pred_final = pred_complete * (1 - mod)
        pred_final = np.round(pred_final).astype(int)
        
        # Extraer solo los días del mes actual
        dias_mes_actual = pred_final[0][-self.days_month:]
        
        # Calcular proyección acumulada
        proyeccion = np.cumsum(dias_mes_actual)
        
        print(f"\nProyección diaria para {grupo_marca}:")
        cadena_pred_diaria = '   '.join([f"{i}-{valor}" for i, valor in enumerate(dias_mes_actual, 1)])
        print(f"[ {cadena_pred_diaria} ]")
        
        print(f"\nProyección acumulada para {grupo_marca}:")
        cadena_proy = '   '.join([f"{i}-{valor}" for i, valor in enumerate(proyeccion, 1)])
        print(f"[ {cadena_proy} ]")
        
        # Interacción con el usuario para confirmar o modificar la proyección
        self.gestionar_confirmacion_proyeccion(grupo_marca, proyeccion, dias_mes_actual, scaler, predicciones_norm, bdproy_grupo)

    def ensure_31_days(self, data):
        """Asegura que los datos tengan exactamente 31 días"""
        if data.shape[1] < 31:
            padding = 31 - data.shape[1]
            data = np.hstack([np.zeros((data.shape[0], padding), dtype=np.float32), data])
        return data[:, -31:].astype(np.float32)

    def gestionar_confirmacion_proyeccion(self, grupo_marca, proyeccion, proyeccion_diaria, scaler, pred_norm, bdproy_grupo):
        """Gestiona la interacción con el usuario para confirmar o modificar la proyección"""
        proyeccion_diaria = proyeccion_diaria.copy()  # Hacer una copia para modificaciones
        
        while True:
            print("\n" + "-"*50)
            respuesta = input(f"¿Desea actualizar la proyección para {grupo_marca}? (S/N o ingrese formato DIA-VALOR para modificar): ")
            
            if respuesta.upper() == 'S':
                print(f"Actualizando proyección para {grupo_marca}...")
                self.actualizar_proyecciones(self.current_per, proyeccion, bdproy_grupo)
                print(f"Proyección actualizada para {grupo_marca}")
                break
            elif respuesta.upper() == 'N':
                print(f"Proyección NO actualizada para {grupo_marca}")
                break
            else:
                try:
                    # Intentar procesar correcciones múltiples en formato DIA-VALOR
                    tokens = respuesta.split()
                    aplicado = False
                    for token in tokens:
                        try:
                            dia_str, val_str = token.split('-', 1)
                            dia = int(dia_str)
                            if dia < 1 or dia > len(proyeccion_diaria):
                                print(f"  • Día fuera de rango: {dia}")
                                continue
                            valor = float(val_str)
                            proyeccion_diaria[dia-1] = valor
                            aplicado = True
                            print(f"  • Día {dia} actualizado a {valor}")
                        except ValueError:
                            print(f"  • Formato inválido: '{token}'")

                    if not aplicado:
                        print("No se aplicaron cambios. Use el formato DIA-VALOR, por ejemplo: 20-40 22-70")
                    else:
                        # Recalcular la proyección acumulada
                        nueva_proyeccion = np.cumsum(proyeccion_diaria)
                        print(f"\nArray diario actualizado para {grupo_marca}:")
                        cadena_proyeccion = '   '.join(f"{i}-{int(v)}" for i, v in enumerate(proyeccion_diaria, 1))
                        print(f"[ {cadena_proyeccion} ]")
        
                        print(f"\nNueva proyección acumulada para {grupo_marca}:")
                        cadena_proyeccion = '   '.join(f"{i}-{int(v)}" for i, v in enumerate(nueva_proyeccion, 1))
                        print(f"[ {cadena_proyeccion} ]")
                        # Actualizar proyección
                        proyeccion = nueva_proyeccion

                except ValueError:
                    print("Entrada no válida. Ingrese 'S', 'N' o un número entero.")

    def obtener_datos_proyeccion(self, grupo_marca):
        """Obtiene los datos de proyección de la API"""
        url = f'https://mafperu.com.pe/wa_data_comercial/GetDataProyeccionMatriz?Grupo={grupo_marca}'
        try:
            respuesta = requests.get(url, timeout=30)
            if respuesta.status_code == 200:
                datos = respuesta.json()
                df = pd.DataFrame(datos)
                if df.empty:
                    print(f"No hay datos disponibles para {grupo_marca}")
                    return None
                return df
            else:
                print(f"Error al obtener los datos de proyección para {grupo_marca}: {respuesta.status_code}")
                return None
        except Exception as e:
            print(f"Error de conexión para {grupo_marca}: {e}")
            return None

    def cargar_scaler(self, scaler_grupo):
        """Carga el scaler para normalización de datos"""
        try:
            scaler_path = f'modelos/scaler_act_{scaler_grupo}.joblib'
            print(f'Cargando scaler: {scaler_path}')
            return load(scaler_path)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo scaler para {scaler_grupo}")
            print(f"Asegúrese de que existe: modelos/scaler_act_{scaler_grupo}.joblib")
            return None

    def cargar_modelo_pytorch(self, scaler_grupo, N):
        """Carga el modelo PyTorch entrenado"""
        try:
            model_path = f'modelos/best_{scaler_grupo}_{N}.pth'
            print(f'Cargando modelo PyTorch: {model_path}')
            
            # Crear instancia del modelo
            model = SequentialLSTM().to(device)
            
            # Cargar los pesos
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            
            print(f"Modelo cargado exitosamente para N={N}")
            return model
            
        except FileNotFoundError:
            print(f"Error: No se encontró el modelo para {scaler_grupo} con N={N}")
            print(f"Buscando modelos alternativos...")
            
            # Buscar modelos con valores de N cercanos
            for alt_N in range(max(1, N-2), min(31, N+3)):
                try:
                    alt_model_path = f'modelos/best_{scaler_grupo}_seq31_n_{alt_N}.pth'
                    if os.path.exists(alt_model_path):
                        print(f"Usando modelo alternativo con N={alt_N}")
                        model = SequentialLSTM().to(device)
                        model.load_state_dict(torch.load(alt_model_path, weights_only=True, map_location=device))
                        return model
                except:
                    continue
                    
            print(f"No se encontró ningún modelo válido para {scaler_grupo}")
            return None
            
        except Exception as e:
            print(f"Error al cargar el modelo PyTorch: {e}")
            return None

    def actualizar_proyecciones(self, periodo, proyeccion, nombre_marca):
        """Actualiza las proyecciones en la API"""
        url = 'https://mafperu.com.pe/wa_data_comercial/PutProyeccion'
        
        for indice, proy in enumerate(proyeccion, start=1):
            dia = indice
            try:
                respuesta = self.update_proy(url, periodo, dia, 'Colocaciones', nombre_marca, 'Nuevo', proy)
                print(f'Respuesta para {nombre_marca}, Proy {proy} en el día {dia}: {respuesta.status_code}')
            except Exception as e:
                print(f'Error actualizando día {dia} para {nombre_marca}: {e}')

    def update_proy(self, url, periodo, dia, estado_sol, marca, estado_veh, proy):
        """Realiza la actualización de una proyección individual"""
        dia = int(dia)
        proy = int(proy)
        datos = {
            'periodo': periodo,
            'dia': dia,
            'EstadoSol': estado_sol,
            'Marca': marca,
            'EstadoVeh': estado_veh,
            'Proy': proy
        }
        respuesta = requests.post(url, json=datos, timeout=30)
        return respuesta

    def listar_modelos_disponibles(self):
        """Lista todos los modelos disponibles en el directorio"""
        print("\n=== MODELOS DISPONIBLES ===")
        modelos_dir = 'modelos'
        if not os.path.exists(modelos_dir):
            print("Directorio 'modelos' no encontrado")
            return
            
        archivos = os.listdir(modelos_dir)
        scalers = [f for f in archivos if f.startswith('scaler_act_') and f.endswith('.joblib')]
        modelos = [f for f in archivos if f.startswith('best_') and f.endswith('.pth')]
        
        print("Scalers disponibles:")
        for scaler in sorted(scalers):
            print(f"  - {scaler}")
            
        print("\nModelos disponibles:")
        for modelo in sorted(modelos):
            print(f"  - {modelo}")


# Ejemplo de uso
if __name__ == "__main__":
    proyector = ProyeccionGrupo()
    
    # Opcional: listar modelos disponibles
    proyector.listar_modelos_disponibles()
    
    # Ejecutar proyecciones
    proyector.ejecutar_proyecciones()