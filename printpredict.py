import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# ==== 1) Carga y prepro ====
os.makedirs('modelos', exist_ok=True)

url = 'https://mafperu.com.pe/wa_data_comercial/GetDataProyeccionMatriz?Grupo=TOYOTA'
r = requests.get(url, timeout=300)
r.raise_for_status()
df = pd.DataFrame(r.json())

# Asegura tipos y filtra (opcional si el API ya entrega todo limpio)
df['periodo'] = df['periodo'].astype(str)
current_per = datetime.now().strftime("%Y%m")

df_act = df[(df['periodo'] != current_per) & (df['tipo'] == 'acts')]
df_dom = df[(df['periodo'] != current_per) & (df['tipo'] == 'doms')]

def ensure_31_days(data):
    if data.shape[1] < 31:
        padding = 31 - data.shape[1]
        data = np.hstack([np.zeros((data.shape[0], padding), dtype=np.float32), data])
    return data[:, -31:].astype(np.float32)

acts = ensure_31_days(np.array(df_act))
doms = ensure_31_days(np.array(df_dom))


doms = np.nan_to_num(doms, nan=0.0)
doms = (doms >= 0.5).astype(np.float32)

# Escalar SOLO acts
scaler_act = StandardScaler()
acts_norm = scaler_act.fit_transform(acts.astype(np.float32))
dump(scaler_act, 'modelos/scaler_act_tdp.joblib')

# ==== 2) Dataset para salida (31) con pérdida enmascarada ====
# Para cada N: input = [acts_parciales(31), doms(31)] -> output = acts(31)
# Entrenamos calculando la pérdida SOLO en los días N..30 (31-N días)

from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

M = acts_norm.shape[0]  # muestras
T = 31                  # timesteps fijos
F = 2                   # canales: acts_parciales y doms

for N in range(9, 12):  # ajusta N como quieras
    print(f"Entrenando con ventana N={N}")

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
    
    # ==== 3) Modelo: salida secuencial (31,1) ====
    model = Sequential([
        # input_shape=(timesteps, features) = (31, 2)
        Bidirectional(LSTM(64, activation='tanh', return_sequences=True), input_shape=(T, F)),
        BatchNormalization(),
        Dropout(0.1),

        LSTM(32, activation='tanh', return_sequences=True),
        BatchNormalization(),
        Dropout(0.05),

        TimeDistributed(Dense(64, activation='relu')),
        Dropout(0.05),
        TimeDistributed(Dense(1, activation='linear'))  # acts estandarizados -> salida lineal
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=Huber(delta=1.0), metrics=['mae'])

    # Callbacks
    ckpt_path = f'modelos/best_toyota_seq31_n_{N}.keras'
    checkpoint = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)

    # Entrenamiento con máscara temporal (loss sólo en N..30)
    model.fit(
        X, y,
        sample_weight=mask,              # << clave: pérdida enmascarada por timestep
        epochs=3000,
        batch_size=8,
        validation_split=0.1,
        shuffle=True,                    # meses independientes -> ok
        verbose=0,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    # ==== Métricas en validación (solo días N..30, en escala original) ====

    val_split = 0.1
    M_total = X.shape[0]
    val_size = int(np.floor(M_total * val_split))
    if val_size == 0:
        print("No hay suficientes muestras para calcular validación.")
    else:
        train_size = M_total - val_size

        # Keras con validation_split toma el ÚLTIMO 10% como validación
        X_val = X[train_size:]
        y_val_norm = y[train_size:, :, 0]       # (val_size, 31)
        val_mask = mask[train_size:]            # (val_size, 31) 0..N-1 conocidos, N..30 desconocidos

        # Predicción en validación
        y_pred_norm = model.predict(X_val, verbose=0)[:, :, 0]  # (val_size, 31)

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

        print(f"Validación (N={N}) — días desconocidos (N..30):")
        print(f"  MAE:   {mae:.4f}")
        print(f"  MSE:   {mse:.4f}")
        print(f"  RMSE:  {rmse:.4f}")
        print(f"  R²:    {r2:.4f}")
        print(f"  SMAPE: {smape:.2f}%")
        print(f"  WAPE:  {wape:.2f}%")

# ================== EVALUACIÓN Y GRÁFICO POR PERIODO ==================
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def evaluar_y_graficar_periodo(i, N, model_path=None, titulo_prefix="TOYOTA"):
    """
    i: índice (fila) en las matrices acts y doms (0..M-1)
    N: número de días conocidos (0..30). Se evalúa WAPE y sMAPE en N..30
    model_path: ruta del modelo .keras; si None => 'modelos/best_toyota_seq31_n_{N}.keras'
    titulo_prefix: texto para el título del gráfico
    """
    assert 0 <= i < acts.shape[0], "Índice i fuera de rango."
    assert 0 <= N < 31, "N debe estar entre 0 y 30."

    if model_path is None:
        model_path = f'modelos/best_toyota_seq31_n_{N}.keras'

    # Cargar modelo sin compilar (no se requiere para inferencia)
    model = load_model(model_path, compile=False)

    # ---------- Preparar entrada (31, 2): [acts_parcial_norm, doms]
    acts_i = acts[i].astype(np.float32)      # escala original
    doms_i = doms[i].astype(np.float32)      # 0/1

    # Normalizar todos los acts (para construir la entrada parcial)
    acts_i_norm = scaler_act.transform(acts_i.reshape(1, -1)).astype(np.float32)[0]

    # Acts parciales: conocidos hasta N-1, desconocidos (N..30) a 0 en la entrada
    acts_partial_norm = acts_i_norm.copy()
    acts_partial_norm[N:] = 0.0

    # Tensor de entrada al modelo: (1, 31, 2)
    X_infer = np.stack([acts_partial_norm, doms_i], axis=-1)[np.newaxis, ...]  # (1,31,2)

    # ---------- Predicción
    y_pred_norm = model.predict(X_infer, verbose=0)[0, :, 0]                   # (31,)
    y_pred = scaler_act.inverse_transform(y_pred_norm.reshape(1, -1))[0]       # escala original
    y_true = acts_i                                                             # (31,)

    # ---------- Métricas SOLO en días desconocidos (N..30)
    eps = 1e-8
    mask = np.zeros(31, dtype=bool)
    mask[N:] = True

    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt

    # WAPE y sMAPE (robustas ante ceros en yt)
    wape  = (np.sum(np.abs(err)) / (np.sum(np.abs(yt)) + eps)) * 100.0
    smape = np.mean(2.0 * np.abs(err) / (np.abs(yt) + np.abs(yp) + eps)) * 100.0

    print(f"[Periodo i={i}, N={N}] Métricas en días desconocidos (N..30)")
    print(f"  WAPE:  {wape:.2f}%")
    print(f"  sMAPE: {smape:.2f}%")

    # ---------- Gráfico
    dias = np.arange(1, 32)  # 1..31
    plt.figure(figsize=(10, 5))
    plt.plot(dias, y_true, label="Real (acts)", linewidth=2)
    plt.plot(dias, y_pred, label="Predicción (acts)", linewidth=2, linestyle="--")
    # Línea divisoria entre conocidos y desconocidos
    if N > 0:
        plt.axvline(x=N+0.5, linestyle=":", linewidth=1.5)

    plt.title(f"{titulo_prefix} | i={i} | N={N} | WAPE={wape:.2f}% | sMAPE={smape:.2f}%")
    plt.xlabel("Día del mes (1-31)")
    plt.ylabel("ACTS (escala original)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------- EJEMPLO DE USO -----------------
evaluar_y_graficar_periodo(i=5, N=25)  # Descomenta y ajusta i y N
evaluar_y_graficar_periodo(i=6, N=20)  # Descomenta y ajusta i y N
