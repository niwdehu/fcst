import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# ==== CONFIGURACIÓN DE MARCAS ====
MARCAS = ['TOYOTA', 'OM', 'HINO', 'USADOS-OM','USADOS-TDP']
BASE_URL = 'https://mafperu.com.pe/wa_data_comercial/GetDataProyeccionMatriz?Grupo='

# ==== 1) Funciones auxiliares ====
def ensure_31_days(data):
    if data.shape[1] < 31:
        padding = 31 - data.shape[1]
        data = np.hstack([np.zeros((data.shape[0], padding), dtype=np.float32), data])
    return data[:, -31:].astype(np.float32)

def cargar_datos_marca(marca):
    """Carga y preprocesa datos para una marca específica"""
    print(f"\n=== Cargando datos para {marca} ===")
    
    url = f"{BASE_URL}{marca}"
    try:
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        
        if df.empty:
            print(f"Warning: No hay datos para {marca}")
            return None, None, None
            
        print(f"Datos cargados: {len(df)} registros")
        
    except Exception as e:
        print(f"Error cargando datos para {marca}: {e}")
        return None, None, None
    
    # Preprocesamiento
    df['periodo'] = df['periodo'].astype(str)
    current_per = datetime.now().strftime("%Y%m")
    
    df_act = df[(df['periodo'] != current_per) & (df['tipo'] == 'acts')]
    df_dom = df[(df['periodo'] != current_per) & (df['tipo'] == 'doms')]
    
    if df_act.empty or df_dom.empty:
        print(f"Warning: Datos insuficientes para {marca} (acts: {len(df_act)}, doms: {len(df_dom)})")
        return None, None, None
    
    # Procesar acts y doms
    acts = ensure_31_days(np.array(df_act.iloc[:, 1:]))  # Asumiendo que primera columna es 'periodo'
    doms = ensure_31_days(np.array(df_dom.iloc[:, 1:]))
    
    doms = np.nan_to_num(doms, nan=0.0)
    doms = (doms >= 0.5).astype(np.float32)
    
    print(f"Forma final - Acts: {acts.shape}, Doms: {doms.shape}")
    
    # Crear y ajustar escalador
    scaler_act = StandardScaler()
    acts_norm = scaler_act.fit_transform(acts.astype(np.float32))
    
    return acts, doms, acts_norm, scaler_act

# ==== 2) Dataset personalizado para PyTorch ====
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.mask = torch.FloatTensor(mask)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]

# ==== 3) Modelo en PyTorch ====
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

# ==== 4) Función de pérdida con máscara ====
class MaskedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(MaskedHuberLoss, self).__init__()
        self.delta = delta
        self.huber = nn.HuberLoss(delta=delta, reduction='none')
    
    def forward(self, predictions, targets, mask):
        # predictions: (batch, seq_len, 1)
        # targets: (batch, seq_len, 1)
        # mask: (batch, seq_len)
        
        predictions = predictions.squeeze(-1)  # (batch, seq_len)
        targets = targets.squeeze(-1)  # (batch, seq_len)
        
        loss = self.huber(predictions, targets)  # (batch, seq_len)
        masked_loss = loss * mask  # aplicar máscara
        
        # Promedio solo sobre elementos enmascarados
        total_loss = masked_loss.sum()
        total_weight = mask.sum()
        
        return total_loss / (total_weight + 1e-8)

# ==== 5) Funciones de entrenamiento ====
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for X_batch, y_batch, mask_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch, mask_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch, mask_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch, mask_batch)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def entrenar_marca_ventana(marca, N, acts_norm, doms, scaler_act):
    """Entrena un modelo para una marca y ventana específica"""
    print(f"\n--- Entrenando {marca} con ventana N={N} ---")
    
    M = acts_norm.shape[0]  # muestras
    T = 31                  # timesteps fijos
    F = 2                   # canales: acts_parciales y doms
    
    if M < 5:  # Verificar datos mínimos
        print(f"Datos insuficientes para {marca} (solo {M} muestras)")
        return None
    
    # acts parciales (conocidos hasta N-1). Futuro a 0 (placeholder)
    acts_partial = acts_norm.copy()
    acts_partial[:, N:] = 0.0  # desconocidos

    # Entrada multicanal (M, T, 2)
    X = np.stack([acts_partial, doms], axis=-1).astype(np.float32)

    # Objetivo completo (M, T, 1) para habilitar sample_weight temporal
    y = acts_norm[..., np.newaxis].astype(np.float32)

    # Máscara temporal (M, T): 0 en días conocidos [0..N-1], 1 en días a predecir [N..30]
    mask = np.zeros((M, T), dtype=np.float32)
    mask[:, N:] = 1.0
    
    # División train/validation
    val_split = 0.1
    val_size = max(1, int(np.floor(M * val_split)))  # Al menos 1 muestra para validación
    train_size = M - val_size
    
    if train_size < 1:
        print(f"Datos insuficientes para división train/val en {marca}")
        return None
    
    # Datasets
    train_dataset = TimeSeriesDataset(X[:train_size], y[:train_size], mask[:train_size])
    val_dataset = TimeSeriesDataset(X[train_size:], y[train_size:], mask[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=min(8, train_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(8, val_size), shuffle=False)
    
    # Modelo, criterio y optimizador
    model = SequentialLSTM().to(device)
    criterion = MaskedHuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    # Entrenamiento
    best_val_loss = float('inf')
    patience = 200
    patience_counter = 0
    
    for epoch in range(3000):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        # Early stopping y guardado del mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = f'modelos/best_{marca.lower()}_{N}.pth'
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping en época {epoch}")
            break
    
    # Cargar el mejor modelo para evaluación
    model.load_state_dict(torch.load(f'modelos/best_{marca.lower()}_{N}.pth', weights_only=True))
    
    # ==== Métricas en validación (solo días N..30, en escala original) ====
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X[train_size:]).to(device)
        y_pred_norm = model(X_val_tensor).cpu().numpy()[:, :, 0]  # (val_size, 31)
    
    y_val_norm = y[train_size:, :, 0]       # (val_size, 31)
    val_mask = mask[train_size:]            # (val_size, 31)
    
    # Inversa del escalado a unidades reales
    y_val = scaler_act.inverse_transform(y_val_norm)        # (val_size, 31)
    y_pred = scaler_act.inverse_transform(y_pred_norm)      # (val_size, 31)
    
    # Aplanar SOLO los días desconocidos (N..30)
    mask_bool = val_mask.astype(bool)
    yt = y_val[mask_bool]
    yp = y_pred[mask_bool]
    
    # Métricas
    eps = 1e-8
    err = yp - yt
    
    mae  = np.mean(np.abs(err))
    mse  = np.mean(err**2)
    rmse = np.sqrt(mse)
    ss_res = np.sum(err**2)
    ss_tot = np.sum((yt - yt.mean())**2) + eps
    r2 = 1.0 - ss_res / ss_tot
    
    smape = np.mean(2.0 * np.abs(err) / (np.abs(yt) + np.abs(yp) + eps)) * 100.0
    wape  = (np.sum(np.abs(err)) / (np.sum(np.abs(yt)) + eps)) * 100.0
    
    print(f"Validación {marca} (N={N}) — días desconocidos (N..30):")
    print(f"  MAE:   {mae:.4f}")
    print(f"  MSE:   {mse:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  R²:    {r2:.4f}")
    print(f"  SMAPE: {smape:.2f}%")
    print(f"  WAPE:  {wape:.2f}%")
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'smape': smape,
        'wape': wape,
        'model_path': f'modelos/best_{marca.lower()}_seq31_n_{N}.pth'
    }

# ==== 6) ENTRENAMIENTO PRINCIPAL MULTI-MARCA ====
def main():
    # Crear directorio de modelos
    os.makedirs('modelos', exist_ok=True)
    
    # Resultados globales
    resultados_globales = {}
    
    # Procesar cada marca
    for marca in MARCAS:
        print(f"\n{'='*60}")
        print(f"PROCESANDO MARCA: {marca}")
        print(f"{'='*60}")
        
        # Cargar datos de la marca
        datos_marca = cargar_datos_marca(marca)
        if datos_marca[0] is None:  # Si no se pudieron cargar los datos
            print(f"Saltando {marca} por problemas en carga de datos")
            continue
            
        acts, doms, acts_norm, scaler_act = datos_marca
        
        # Guardar el escalador específico de la marca
        scaler_path = f'modelos/scaler_act_{marca.lower()}.joblib'
        dump(scaler_act, scaler_path)
        print(f"Escalador guardado: {scaler_path}")
        
        # Resultados para esta marca
        resultados_marca = {}
        
        # Entrenar para diferentes ventanas N
        for N in range(2, 31):  # Ajusta según necesites
            try:
                resultado = entrenar_marca_ventana(marca, N, acts_norm, doms, scaler_act)
                if resultado:
                    resultados_marca[N] = resultado
            except Exception as e:
                print(f"Error entrenando {marca} N={N}: {e}")
                continue
        
        resultados_globales[marca] = resultados_marca
        
        # Resumen para la marca
        if resultados_marca:
            print(f"\n--- RESUMEN {marca} ---")
            print(f"Modelos entrenados: {len(resultados_marca)}")
            wapes = [r['wape'] for r in resultados_marca.values()]
            smapes = [r['smape'] for r in resultados_marca.values()]
            print(f"WAPE promedio: {np.mean(wapes):.2f}% ± {np.std(wapes):.2f}%")
            print(f"sMAPE promedio: {np.mean(smapes):.2f}% ± {np.std(smapes):.2f}%")
    
    # ==== RESUMEN GLOBAL ====
    print(f"\n{'='*60}")
    print("RESUMEN GLOBAL DEL ENTRENAMIENTO")
    print(f"{'='*60}")
    
    for marca, resultados in resultados_globales.items():
        if resultados:
            print(f"\n{marca}:")
            print(f"  Modelos: {len(resultados)}")
            wapes = [r['wape'] for r in resultados.values()]
            smapes = [r['smape'] for r in resultados.values()]
            print(f"  WAPE: {np.mean(wapes):.2f}% ± {np.std(wapes):.2f}%")
            print(f"  sMAPE: {np.mean(smapes):.2f}% ± {np.std(smapes):.2f}%")
            
            # Mejor modelo para esta marca
            mejor_n = min(resultados.keys(), key=lambda n: resultados[n]['wape'])
            print(f"  Mejor N: {mejor_n} (WAPE: {resultados[mejor_n]['wape']:.2f}%)")
    
    print(f"\nEntrenamiento completado. Archivos guardados en 'modelos/'")
    return resultados_globales

if __name__ == "__main__":
    resultados = main()