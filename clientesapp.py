# Importamos nuestras herramientas
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Título
st.title("Predicción de Clientes de Alto Valor")
#Subtítulo
st.markdown("Esta aplicación predice la probabilidad de que un Cliente sea de alto valor para la empresa.")

#importamos nuestro data set modelo
df = pd.read_csv(
    "C:\\Users\\USUARIO\\OneDrive\\Documentos\\ciencia de datos\\proyectos de ciencia de datos\\MLredneuronal\\Clientesweb\\data\\Clientes-RedesN.csv"
)

# Separamos variables
X = df[["edad", "ingresos", "gasto"]]
y = df["clase"]

# dividimos datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# smote para balancear clases del dataset
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


# Balanceo de clases para que el modelo preste atención a la clase minoritaria
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_res),
    y=y_train_res
)
class_weight_dict = dict(enumerate(class_weights))

# Escalar datos
scaler = MinMaxScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

#Red Neuronal
model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Entrenar
model.fit(
    X_train_res_scaled,
    y_train_res,
    epochs=300,
    batch_size=8,
    verbose=0,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stop]
)

# Umbral de clasificación óptimo
y_test_prob = model.predict(X_test_scaled).flatten()
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

#Evaluacion del modelo
st.subheader("📈 Métricas de Desempeño")

col1, col2, col3 = st.columns(3)
with col1:
    auc_score = roc_auc_score(y_test, y_test_prob)
    st.metric("ROC-AUC Score", f"{auc_score:.3f}")
with col2:
    st.metric("Umbral Óptimo", f"{optimal_threshold:.3f}")
with col3:
    st.metric("Umbral Actual", "Ajustable ⚙️")

# Predicción con umbral óptimo
y_test_pred_optimal = (y_test_prob > optimal_threshold).astype(int)


#Entrada de usuario
st.sidebar.header("🔍 Ingresa los datos del cliente")
def user_input_features():
    edad = st.sidebar.number_input("Edad", min_value=18, max_value=100, value=30)
    ingresos = st.sidebar.number_input("Ingresos mensuales", min_value=10000, max_value=100000, step=1000, value=50000)
    gasto = st.sidebar.number_input("Gasto promedio mensual", min_value=1000, max_value=100000, step=500, value=20000)
    return pd.DataFrame({"edad": [edad], "ingresos": [ingresos], "gasto": [gasto]})

input_df = user_input_features()

# Umbral ajustable para el modelo
st.sidebar.header("⚙️ Configuración del umbral")
st.sidebar.info(f"💡 Umbral óptimo sugerido: {optimal_threshold:.3f}")
umbral = st.sidebar.slider(
    "Probabilidad mínima para considerar ALTO valor",
    0.0, 1.0, float(optimal_threshold), 0.01
)


# Predicción
if st.sidebar.button("🚀 Predecir", type="primary"):
    input_scaled = scaler.transform(input_df)
    
    # Predicción
    pred_result = model.predict(input_scaled)
    probabilidad = pred_result[0][0]
    probabilidad = probabilidad.item()
    prediccion = 1 if probabilidad > umbral else 0

    st.subheader("🎯 Resultado de la Predicción")
    
    if prediccion == 1:
        st.success(f"✅ Cliente de **ALTO VALOR**")
    else:
        st.error(f"❌ Cliente de **BAJO VALOR**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probabilidad", f"{probabilidad:.2%}")
    with col2:
        st.metric("Umbral Usado", f"{umbral:.2%}")
    

#Datos originales
with st.expander("📋 Ver muestra del Dataset"):
    st.write(df.head(20))