import deepxde as dde
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math
from deepxde.backend import tf
import streamlit as st
from scipy.integrate import solve_ivp
import re

def get_data():
   # Datos del tiempo (t) en días
   t = np.array([3.46, 4.58, 5.67, 6.64, 7.63, 8.41, 9.32, 10.27, 11.19, 12.39, 13.42, 15.19, 16.24, 17.23, 18.18, 19.29,
                 21.23, 21.99, 24.33, 25.58, 26.43, 27.44, 28.43, 30.49, 31.34, 32.34, 33.0, 35.2, 36.34, 37.29, 38.5, 39.67,
                 41.37, 42.58, 45.39, 46.38, 48.29, 49.24, 50.19, 51.14, 52.10, 54.0, 56.33, 57.33, 59.38])
   # Volumen de las células cancerígenas 10^9 νm3
   V = np.array([0.0158, 0.0264, 0.0326, 0.0445, 0.0646, 0.0933, 0.1454, 0.2183, 0.2842, 0.4977, 0.6033, 0.8441, 1.2163, 1.447, 2.3298,
                 2.5342, 3.0064, 3.4044, 3.2046, 4.5241, 4.3459, 5.1374, 5.5376, 4.8946, 5.0660, 6.1494, 6.8548, 5.9668, 6.6945, 6.6395,
                 6.8971, 7.2966, 7.2268, 6.8815, 8.0993, 7.2112, 7.0694, 7.4971, 6.9974, 6.7219, 7.0523, 7.1095, 7.0694, 8.0562, 7.2268])
   return pd.DataFrame({'t': t, 'V': V})

# Extracción de datos
df = get_data()

# Ejecución del modelo
def run_model(model="Verhulst", vars={'k': 25.0, 'C': 25.0, 'theta': 25.0}, lr=0.001, iters=5000):
   # Inicialización de la función de puntos dentro y fuera del subdominio de condiciones iniciales
   def boundary(_, on_initial):
      return on_initial
   
   # Dominio del problema (días: [3.46, 50.38])
   geom = dde.geometry.TimeDomain(df['t'].min(), df['t'].max())
   
   # Condición inicial
   ic1 = dde.icbc.IC(geom, lambda X: float(df['V'].values[0]), boundary)
   
   # Organización y asignación de los datos de entrenamiento
   observe_t = df['t'].values
   observe_V = df['V'].values
   observe_Ys = dde.icbc.PointSetBC(observe_t.reshape(-1, 1), observe_V.reshape(-1, 1), component=0)

   # Definición de las variables para ambos modelos
   k = dde.Variable(vars['k'])
   C = dde.Variable(vars['C'])

   if model == "Verhulst":
      
      def verhulst(t, v):
         # Función que representa la ecuación diferencial a resolver, con v (volumen) = p(t) (población de células en tiempo t)
         dpdt = dde.grad.jacobian(v,t, i=0, j=0)
         return dpdt - k*v*(1 - v/C)
      
      # Arquitectura de la red neuronal: 3 capas ocultas con 50 neuronas cada una y activación tanh
      layer_size = [1] + [50]*3 + [1]
      activation = "tanh"
      initializer = "Glorot uniform"
      net = dde.nn.FNN(layer_size, activation, initializer)

      # Definición de los datos del problema y los parámetros a hallar
      data = dde.data.PDE(geom, verhulst, [ic1, observe_Ys], num_domain=400, num_boundary=2, anchors=observe_t.reshape(-1, 1))
      external_trainable_variables = [k, C]
      variable = dde.callbacks.VariableValue(external_trainable_variables, period=100, filename="variables.dat")
      
      # Creación y compilación del modelo con red, datos y optimizador Adam
      model = dde.Model(data, net)

      # train adam
      model.compile("adam", lr=lr, external_trainable_variables=external_trainable_variables)

      losshistory, train_state = model.train(iterations=iters, callbacks=[variable])

      # Puntos de prueba en el dominio para comparar la predicción del modelo con la solución exacta
      x_test = geom.uniform_points(45, True)
      y_pred = model.predict(x_test)

      # Se regresan el modelo, las predicciones, la pérdida del entrenamiento y las variables halladas
      return model, x_test, y_pred, losshistory, train_state, model.sess.run(k), model.sess.run(C)
   
   if model == "Montroll":
      # Definición de theta para Montroll
      theta = dde.Variable(vars['theta'])

      def montroll(t, v):
         # Función que representa la ecuación diferencial a resolver, con v (volumen) = p(t) (población de células en tiempo t)
         dpdt = dde.grad.jacobian(v,t, i=0, j=0)
         v_transformed = dde.backend.tf.nn.softplus(v)
         return dpdt - k*v_transformed*(1 - (v_transformed/C)**theta)

      # Arquitectura de la red neuronal: 3 capas ocultas con 50 neuronas cada una y activación tanh
      layer_size = [1] + [50]*3 + [1]
      activation = "tanh"
      initializer = "Glorot uniform"
      net = dde.nn.FNN(layer_size, activation, initializer)

      # Definición de los datos del problema y los parámetros a hallar
      data = dde.data.PDE(geom, montroll, [ic1, observe_Ys], num_domain=400, num_boundary=2, anchors=observe_t.reshape(-1, 1))
      external_trainable_variables = [k, C, theta]
      variable = dde.callbacks.VariableValue(external_trainable_variables, period=100, filename="./variables.dat")
      
      # Creación y compilación del modelo con red, datos y optimizador Adam
      model = dde.Model(data, net)

      # train adam
      model.compile("adam", lr=lr, external_trainable_variables=external_trainable_variables)
      losshistory, train_state = model.train(iterations=iters, callbacks=[variable])

      # Puntos de prueba en el dominio para comparar la predicción del modelo con la solución exacta
      x_test = geom.uniform_points(45, True)
      y_pred = model.predict(x_test)

      # Se regresan el modelo, las predicciones, la pérdida del entrenamiento y las variables halladas
      return model, x_test, y_pred, losshistory, train_state, model.sess.run(k), model.sess.run(C), model.sess.run(theta)
   
def get_scipy_vals(df=pd.DataFrame(), model_eq="Verhulst", vars={'k': 0.1, 'C': 7, 'theta': 0.2}):
   # SciPy Solution
   # Se define el dominio, y la p(0) inicial
    p0 = [float(df['V'].values[0])]
    t_span = (float(df['t'].min()), float(df['t'].max()))
    t_eval = np.linspace(t_span[0], t_span[1], 45)

    # Variables a usar
    k = vars['k']
    C = vars['C']

    # Ecuación diferencial de cada modelo
    if model_eq == "Verhulst":
       def eq(t, p):
          return k * p * (1 - p / C)
    if model_eq == "Montroll":
       theta = vars['theta']
       def eq(t, p):
          return k * p * (1 - (p/C)**theta)
       
       # Solución de la ecuación en el dominio, con las condiciones y los parámetros obtenidos
    sol_paper = solve_ivp(eq, t_span, p0, t_eval=t_eval)
    return sol_paper.t, sol_paper.y[0]

def visualize_test(x_test, y_pred, x_scipy, y_scipy):
   # Figura
   fig = go.Figure()
   
   # Datos
   x_values = x_test.flatten()
   y_values = df['V'].values
   t_values = df['t'].values
   y_pred_values = y_pred.flatten()

   # Predicciones realizadas con la solución analítica de scipy
   x_scipy_vals = x_scipy.flatten()
   y_scipy_vals = y_scipy.flatten()
   
   # Configurar la animación para los puntos y las líneas
   frames = []
   for i in range(1, len(x_values) + 1):
      frames.append(go.Frame(
         data=[go.Scatter(x=x_values[:i], y=y_values[:i], mode='markers', name='Real Point',
                          marker=dict(size=t_values[:i]/3, color='cyan')),
               go.Scatter(x=x_values[:i], y=y_pred_values[:i], mode='lines', name='PINN Prediction', 
                          line=dict(color='blue', width=2)),
               go.Scatter(x=x_scipy_vals[:i], y=y_scipy_vals[:i], mode='lines', name='Analytical Solution',
                          line=dict(color='white', width=2))], name=f'frame_{i}'))
   # Visualización inicial (vacía)
   fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Real Point',
                            marker=dict(size=[], color='cyan')))
   fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='PINN Prediction', 
                            line=dict(color='blue', width=2)))
   fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Analytical Solution', 
                            line=dict(color='white', width=2)))
   
   # Configuración de la animación
   fig.update_layout(template='plotly_dark', title='<b>Real Points vs. PINN Prediction</b>', 
                     title_x=0.5, xaxis_title='t (days)', yaxis_title='V (tumor cells volume)', legend_title='Method',  font=dict(size=14), height=500, 
                     width=1000, updatemenus=[dict(type='buttons', showactive=False,
                                                   buttons=[dict(label='Start', method='animate', args=[None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]),
                                                            dict(label='Pause', method='animate', args=[[None], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': False}, 'transition': {'duration': 0}}])],
                                                            direction='left', pad={'r': 10, 't': 10}, x=0.1, y=0.05, xanchor='right',
                                                            yanchor='top')],
   sliders=[dict(steps=[dict(method='animate', args=[[f'frame_{i}'], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True}, 'transition': {'duration': 0}}], label=str(i)) 
                         for i in range(len(frames))], transition={'duration': 0}, x=0.1, y=0,
                         currentvalue={'font': {'size': 14}, 'prefix': 'point: ', 'visible': True, 'xanchor': 'right'}, len=0.9)])
   # Añadir los frames generados
   fig.frames = frames
   
   # Regresa figura
   return fig

def visualize_loss(loss_history):
   # Suma de los componentes de la pérdida para entrenamiento y validación (L_total = L_entrenamiento + L_validación)
   loss_total_train = [np.sum(l) for l in losshistory.loss_train]
   loss_total_test = [np.sum(l) for l in losshistory.loss_test]
   
   # Visualización de ambas pérdidas
   fig2 = go.Figure()

   epochs = [epoch for epoch in range(0, len(loss_total_train)+1)]

   # Configurar la animación para los puntos y las líneas
   frames = []

   for i in range(0, len(loss_total_train)):
      frames.append(go.Frame(data=[go.Scatter(x=epochs[:i+1], y=loss_total_train[:i+1], mode='lines', name='Train Loss', 
                                              line=dict(color='red', width=2)),
                                   go.Scatter(x=epochs[:i+1], y=loss_total_test[:i+1], mode='lines', name='Test Loss',
                                              line=dict(color='crimson', width=2))], name=f'frame_{i}'))
   
   # Visualización inicial (vacía)
   fig2.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Train Loss', 
                            line=dict(color='red', width=2)))
   fig2.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Test Loss', 
                            line=dict(color='crimson', width=2)))
   
   # Configuración de la animación
   fig2.update_layout(template='plotly_dark', title='<b>Train vs. Test Loss</b>', yaxis_type='log',
                      plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                      title_x=0.5, xaxis_title='Epoch × 10<sup>3</sup>', yaxis_title='Loss', legend_title='Train/Test', 
                      font=dict(size=14), height=750, width=500,
                      updatemenus=[dict(type='buttons', showactive=False,
                                        buttons=[dict(label='Start', method='animate', args=[None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]),
                                                 dict(label='Pause', method='animate', args=[[None], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': False}, 'transition': {'duration': 0}}])],
                                                 direction='left', pad={'r': 10, 't': 10}, x=0.1, y=0.05, xanchor='right',
                                                 yanchor='top')],
   sliders=[dict(steps=[dict(method='animate', args=[[f'frame_{i}'], {'mode': 'immediate', 'frame': {'duration': 0.25, 'redraw': True}, 'transition': {'duration': 0.5}}], label=str(i)) 
                         for i in range(len(frames))], transition={'duration': 0.5}, x=0.1, y=0,
                         currentvalue={'font': {'size': 14}, 'prefix': 'epoch: ', 'visible': True, 'xanchor': 'right'}, len=0.9)])
   # Añadir los frames generados
   fig2.frames = frames
   
   # Regresa figura
   return fig2

def visualize_vars(vars_df=pd.DataFrame(), used_model="Verhulst"):
   # Visualización de ambas pérdidas
   fig_k, fig_C = go.Figure(), go.Figure()
   epochs = [epoch for epoch in range(len(vars_df))]

   # Configurar la animación para los puntos y las líneas
   frames_k, frames_C = [], []
   if used_model == "Montroll":
      fig_theta = go.Figure()
      frames_theta = []

   for i in range(len(vars_df)):
      y_k = list(vars_df["k"].values)
      y_C = list(vars_df["C"].values)
      frames_k += [go.Frame(data=[go.Scatter(x=epochs[:i+1], y=y_k[:i+1], mode='lines', name='k', 
                                             line=dict(color='red', width=2))], name=f'frame_{i}')]
      frames_C += [go.Frame(data=[go.Scatter(x=epochs[:i+1], y=y_C[:i+1], mode='lines', name='C',
                                             line=dict(color='blue', width=2))], name=f'frame_{i}')]
      if used_model == "Verhulst":
         continue
      if used_model == "Montroll":
         y_theta = list(vars_df["theta"].values)
         frames_theta += [go.Frame(data=[go.Scatter(x=epochs[:i+1], y=y_theta[:i+1], mode='lines', name='θ',
                                                    line=dict(color='white', width=2))], name=f'frame_{i}')]
   
   # Visualización inicial (vacía)
   fig_k.add_trace(go.Scatter(x=[], y=[], mode='lines', name='k', line=dict(color='red', width=2)))
   fig_C.add_trace(go.Scatter(x=[], y=[], mode='lines', name='C', line=dict(color='blue', width=2)))
   if used_model == "Montroll":
      fig_theta.add_trace(go.Scatter(x=[], y=[], mode='lines', name='θ', line=dict(color='white', width=2)))
   
   # Configuración de la animación
   fig_k.update_layout(template='plotly_dark', title='<b>k Evolution per Epoch</b>', yaxis_type='log',
                       plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                       title_x=0.5, xaxis_title='Epoch', yaxis_title='k', legend_title='Variable', 
                       font=dict(size=14), height=750, width=500,
                       updatemenus=[dict(type='buttons', showactive=False,
                                         buttons=[dict(label='Start', method='animate', args=[None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]),
                                                  dict(label='Pause', method='animate', args=[[None], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': False}, 'transition': {'duration': 0}}])],
                                                  direction='left', pad={'r': 10, 't': 10}, x=0.1, y=0.05, xanchor='right',
                                                  yanchor='top')],
                                                  sliders=[dict(steps=[dict(method='animate', args=[[f'frame_{i}'], {'mode': 'immediate', 'frame': {'duration': 0.25, 'redraw': True}, 'transition': {'duration': 0.5}}], label=str(i)) 
                                                                       for i in range(len(frames_k))], transition={'duration': 0.5}, x=0.1, y=0,
                                                                       currentvalue={'font': {'size': 14}, 'prefix': 'epoch × 10<sup>2</sup>: ', 'visible': True, 'xanchor': 'right'}, len=0.9)])
   fig_C.update_layout(template='plotly_dark', title='<b>C Evolution per Epoch</b>', yaxis_type='log',
                       plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                       title_x=0.5, xaxis_title='Epoch', yaxis_title='C', legend_title='Variable', 
                       font=dict(size=14), height=750, width=500,
                       updatemenus=[dict(type='buttons', showactive=False,
                                         buttons=[dict(label='Start', method='animate', args=[None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]),
                                                  dict(label='Pause', method='animate', args=[[None], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': False}, 'transition': {'duration': 0}}])],
                                                  direction='left', pad={'r': 10, 't': 10}, x=0.1, y=0.05, xanchor='right',
                                                  yanchor='top')],
                                                  sliders=[dict(steps=[dict(method='animate', args=[[f'frame_{i}'], {'mode': 'immediate', 'frame': {'duration': 0.25, 'redraw': True}, 'transition': {'duration': 0.5}}], label=str(i)) 
                                                                       for i in range(len(frames_k))], transition={'duration': 0.5}, x=0.1, y=0,
                                                                       currentvalue={'font': {'size': 14}, 'prefix': 'epoch × 10<sup>2</sup>: ', 'visible': True, 'xanchor': 'right'}, len=0.9)])
   # Añadir los frames generados
   fig_k.frames = frames_k
   fig_C.frames = frames_C
   if used_model == "Verhulst":
      return fig_k, fig_C
   if used_model == "Montroll":
      fig_theta.update_layout(template='plotly_dark', title='<b>θ Evolution per Epoch</b>', yaxis_type='log',
                              plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                              title_x=0.5, xaxis_title='Epoch', yaxis_title='θ', legend_title='Variable', 
                              font=dict(size=14), height=750, width=500,
                              updatemenus=[dict(type='buttons', showactive=False,
                                                buttons=[dict(label='Start', method='animate', args=[None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]),
                                                         dict(label='Pause', method='animate', args=[[None], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': False}, 'transition': {'duration': 0}}])],
                                                         direction='left', pad={'r': 10, 't': 10}, x=0.1, y=0.05, xanchor='right',
                                                         yanchor='top')],
                                                         sliders=[dict(steps=[dict(method='animate', args=[[f'frame_{i}'], {'mode': 'immediate', 'frame': {'duration': 0.25, 'redraw': True}, 'transition': {'duration': 0.5}}], label=str(i)) 
                                                                              for i in range(len(frames_k))], transition={'duration': 0.5}, x=0.1, y=0,
                                                                              currentvalue={'font': {'size': 14}, 'prefix': 'epoch × 10<sup>2</sup>: ', 'visible': True, 'xanchor': 'right'}, len=0.9)])
      fig_theta.frames = frames_theta
      return fig_k, fig_C, fig_theta
       
# Background (CSS)
bg = """
<style>
@keyframes move {
100% {
transform: translate3d(0, 0, 1px) rotate(360deg);
}
}
       
.background {
position: fixed;
width: 100vw;
height: 100vh;
top: 0;
left: 0;
background: #000000;
overflow: hidden;
}
       
.ball {
position: absolute;
width: 20vmin;
height: 20vmin;
border-radius: 50%;
backface-visibility: hidden;
animation: move linear infinite;
}
       
.ball:nth-child(odd) {
color: #051094;
}
       
.ball:nth-child(even) {
color: #82EEFD;
}
       
/* Using a custom attribute for variability */
.ball:nth-child(1) {
top: 77%;
left: 88%;
animation-duration: 40s;
animation-delay: -3s;
transform-origin: 16vw -2vh;
box-shadow: 40vmin 0 5.703076368487546vmin currentColor;
}
       
.ball:nth-child(2) {
top: 42%;
left: 2%;
animation-duration: 53s;animation-delay: -29s;
transform-origin: -19vw 21vh;
box-shadow: -40vmin 0 5.17594621519026vmin currentColor;
}
       
.ball:nth-child(3) {
top: 28%;
left: 18%;
animation-duration: 49s;
animation-delay: -8s;
transform-origin: -22vw 3vh;
box-shadow: 40vmin 0 5.248179047256236vmin currentColor;
}
       
.ball:nth-child(4) {
top: 50%;
left: 79%;
animation-duration: 26s;
animation-delay: -21s;
transform-origin: -17vw -6vh;
box-shadow: 40vmin 0 5.279749632220298vmin currentColor;
}
       
.ball:nth-child(5) {
top: 46%;
left: 15%;
animation-duration: 36s;
animation-delay: -40s;
transform-origin: 4vw 0vh;
box-shadow: -40vmin 0 5.964309466052033vmin currentColor;
}
       
.ball:nth-child(6) {
top: 77%;
left: 16%;
animation-duration: 31s;
animation-delay: -10s;
transform-origin: 18vw 4vh;
box-shadow: 40vmin 0 5.178483653434181vmin currentColor;
}
       
.ball:nth-child(7) {
top: 22%;
left: 17%;
animation-duration: 55s;
animation-delay: -6s;
transform-origin: 1vw -23vh;
box-shadow: -40vmin 0 5.703026794398318vmin currentColor;
}
       
.ball:nth-child(8) {
top: 41%;
left: 47%;
animation-duration: 43s;
animation-delay: -28s;
transform-origin: 25vw -3vh;
box-shadow: 40vmin 0 5.196265905749415vmin currentColor;
}

.ball:nth-child(9) {
top: 30%;
left: 60%;
animation-duration: 38s;
animation-delay: -15s;
transform-origin: 20vw 10vh;
box-shadow: 40vmin 0 5.5vmin currentColor;
}
       
.ball:nth-child(10) {
top: 65%;
left: 35%;
animation-duration: 45s;
animation-delay: -25s;
transform-origin: -15vw 15vh;
box-shadow: -40vmin 0 5.2vmin currentColor;
}
</style>

<div class="background">
<!-- Using common classes to minimize redundancy -->
<span class="ball"></span>
<span class="ball"></span>
<span class="ball"></span>
<span class="ball"></span>
<span class="ball"></span>
<span class="ball"></span>
<span class="ball"></span>
<span class="ball"></span>
<span class="ball"></span>
<span class="ball"></span>
</div>
"""

# Definición del fondo de la app
st.markdown(bg, unsafe_allow_html=True)

# Título de la app
st.title("Physics-Informed Neural Networks (PINNs) for Tumor Cell Growth Modeling")

# Selección de parámetros para el modelo (modelo, epochs, LR)
selected_model = st.selectbox("Choose the model to solve the problem:", ["", "Verhulst", "Montroll"])

iters = st.slider("Epochs. **Note**: the more epochs, the more it takes for the model to be trained:",
                  5000, 20000, step=1000, value=5000, key="epochs")

lr = st.number_input("Learning rate:", 0.001, 0.01, step=0.001, value=None, placeholder="Type the model's learning rate...", key="LR")

if selected_model == "Verhulst":
   # Ecuación y variables del modelo de Verhulst
   st.header("Verhulst Model")
   st.write("The model is set to solve the following equation:")
   st.latex(r"\frac{dp}{dt}(t)=kp(t)(1-\frac{p(t)}{C})")
   st.write("where:")
   st.markdown(
      R"$p(t)$"
      ": population size of cells at time "
      R"$t$")
   st.markdown(
      R"$t$"
      ": time (days)")
   st.markdown(
      R"$k$"
      ": growth rate")
   st.markdown(
      R"$C$"
      ": carrying capacity")
   
   # Definición de las variables k y C
   k_slider = st.slider("k:", 0.1, 10.0, step=0.1, value=2.5, key="k_slider_verhulst")
   C_slider = st.slider("C:", 0.1, 10.0, step=0.1, value=2.5, key="C_slider_verhulst")
   
   # Se ejecuta solo hasta definir una learning rate (entre 0.001 y 0.01)
   if lr is not None:
      with st.spinner("Training model...", show_time=True):
         # Obtención del modelo
         model, x_test, y_pred, losshistory, train_state, final_k, final_C = run_model(model="Verhulst", vars={'k': k_slider, 'C': C_slider},
                                                                                       lr=lr, iters=iters)
      
      st.success("Model successfully trained!")

      # Lectura y tratamiento de las variables y su evolución en el entrenamiento
      vars_df = pd.read_table("variables.dat", sep="\s+", header=None)
      
      # Primer renglón = columnas
      vars_df.columns = vars_df.iloc[0]
      
      # Se renombran las columnas
      vars_df.columns = ["Epoch", "k", "C"]

      # Se convierten todos los valores de str y formato científico a float
      for col in vars_df.columns[1:]:
         vars_df[col] = vars_df[col].astype(str).apply(lambda x: float(re.sub(r"[\[\],]", "", x.strip())))

      # Para evitar errores en las gráficas, se consideran solo a las variables en el entrenamiento (epoch=0:# de epochs seleccionado)
      vars_df = vars_df[(vars_df["Epoch"] >= 0) & (vars_df["Epoch"] <= iters)]

      # Obtención de predicciones con la solución analítica
      x_scipy, y_scipy = get_scipy_vals(df=df, model_eq="Verhulst", vars={'k':final_k, 'C':final_C})

      # Obtención de gráficas de predicciones, pérdidas y evolución de variables
      fig = visualize_test(x_test, y_pred, x_scipy, y_scipy)
      
      fig.update_layout(title_font=dict(size=18, family="Arial", color="white"), uniformtext_minsize=11.5, uniformtext_mode='hide',
                        title={'y':0.95, 'x':0.5, 'xanchor': 'center','yanchor': 'top'},
                        plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
      
      fig.add_annotation(x=0.5, y=1.12, xref='paper', yref='paper', text=f'Parameters<br>k = {final_k:.4f} | C = {final_C:.4f}',
                         showarrow=False, font=dict(size=10, color="white"))
      
      fig2 = visualize_loss(loss_history=losshistory)

      figK, figC = visualize_vars(vars_df=vars_df, used_model="Verhulst")
      
      # Se muestran las gráficas
      st.plotly_chart(fig)
      st.plotly_chart(fig2)
      st.plotly_chart(figK)
      st.plotly_chart(figC)
   else: # Obligatorio definir la tasa de aprendizaje del modelo
      st.error('Missing Learning Rate', icon="🚨")

if selected_model == "Montroll":
   # Ecuación y variables del modelo de Montroll
   st.header("Montroll Model")
   st.write("The model is set to solve the following equation:")
   st.latex(r"\frac{dp}{dt}(t)=kp(t)(1-(\frac{p(t)}{C})^θ)")
   st.write("where:")
   st.markdown(
      R"$p(t)$"
      ": population size of cells at time "
      R"$t$")
   st.markdown(
      R"$t$"
      ": time (days)")
   st.markdown(
      R"$k$"
      ": growth rate")
   st.markdown(
      R"$C$"
      ": carrying capacity")
   st.markdown(
      R"$\theta$"
      ": position of the inflexion point of the growth curve"
   )

   # Definición de las variables k, C y theta
   k_slider = st.slider("k:", 0.1, 10.0, step=0.1, value=2.5, key="k_slider_montroll")
   C_slider = st.slider("C:", 0.1, 10.0, step=0.1, value=2.5, key="C_slider_montroll")
   theta_slider = st.slider("θ:", 0.1, 10.0, step=0.1, value=2.5, key="theta_slider_montroll")

   # Se ejecuta solo hasta definir una learning rate (entre 0.001 y 0.01)
   if lr is not None:
      with st.spinner("Training model...", show_time=True):
         model, x_test, y_pred, losshistory, train_state, final_k, final_C, final_theta = run_model(model="Montroll", vars={'k': k_slider, 'C': C_slider, 'theta': theta_slider},
                                                                                                    lr=lr, iters=iters)
      
      st.success("Model successfully trained!")

      # Lectura y tratamiento de las variables y su evolución en el entrenamiento
      vars_dfM = pd.read_table("variables.dat", sep="\s+", header=None)
      
      # Primer renglón = columnas
      vars_dfM.columns = vars_dfM.iloc[0]
      
      # Se renombran las columnas
      vars_dfM.columns = ["Epoch", "k", "C", "theta"]

      # Se convierten todos los valores de str y formato científico a float
      for col in vars_dfM.columns[1:]:
         vars_dfM[col] = vars_dfM[col].astype(str).apply(lambda x: float(re.sub(r"[\[\],]", "", x.strip())))

      # Para evitar errores en las gráficas, se consideran solo a las variables en el entrenamiento (epoch=0:# de epochs seleccionado)
      vars_dfM = vars_dfM[(vars_dfM["Epoch"] >= 0) & (vars_dfM["Epoch"] <= iters)]

      # Obtención de predicciones con la solución analítica
      x_scipy, y_scipy = get_scipy_vals(df=df, model_eq="Montroll", vars={'k':final_k, 'C':final_C, 'theta': final_theta})

      # Obtención de gráficas de predicciones, pérdidas y evolución de variables
      fig = visualize_test(x_test, y_pred, x_scipy, y_scipy)
      
      fig.update_layout(title_font=dict(size=18, family="Arial", color="white"), uniformtext_minsize=11.5, uniformtext_mode='hide',
                        title={'y':0.95, 'x':0.5, 'xanchor': 'center','yanchor': 'top'}, plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)')
      
      fig.add_annotation(x=0.5, y=1.12, xref='paper', yref='paper',  
                         text=f'Parameters<br>k = {final_k:.4f} | C = {final_C:.4f} | θ = {final_theta:.4f}',
                         showarrow=False, font=dict(size=10, color="white"))
      
      fig2 = visualize_loss(loss_history=losshistory)

      figK, figC, fig_theta = visualize_vars(vars_df=vars_dfM, used_model="Montroll")

      # Se muestran las gráficas
      st.plotly_chart(fig)
      st.plotly_chart(fig2)
      st.plotly_chart(figK)
      st.plotly_chart(figC)
      st.plotly_chart(fig_theta)
   else: # Obligatorio definir la tasa de aprendizaje del modelo
      st.error('Missing Learning Rate', icon="🚨")
