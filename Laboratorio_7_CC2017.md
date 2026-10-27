# CC2017 – Modelación y Simulación
# Laboratorio 7

## Instrucciones

Usted y su grupo forman el equipo asesor de Aula Abierta, una plataforma de video educativo que crece por el boca a boca. El director técnico debe decidir la política de crecimiento de la infraestructura de servidores: cuándo pedir más servidores y cuántos por pedido. Pedir demasiado pronto cuesta dinero, pedir tarde hace que la plataforma se ponga lenta, y una plataforma lenta pierde usuarios. Su equipo debe entregarle una recomendación respaldada por un modelo híbrido, y debe poder defenderla ante él cuando le pregunte por los límites del modelo. De esa recomendación depende cuántos usuarios conserva la plataforma y cuánto gasta en servidores.

El modelo combina dos paradigmas. Los usuarios activos evolucionan con System Dynamics y el clúster de servidores se simula con eventos discretos. Ambos se acoplan en los dos sentidos.

**Capa SD (usuarios activos, en unidades de usuarios, con el tiempo en días).** Sea $U(t)$ el número de usuarios activos:

$$\frac{dU}{dt} = \left(p + q\frac{U}{M}\right)(M - U) - \delta\, U, \qquad \delta = \delta_0 + \kappa_W \max(0,\, W - W_{\text{ok}}) + \kappa_R\, r$$

El primer término es la adopción por innovación y por boca a boca, como en el modelo de Bass de la semana 3. El segundo es el abandono, que aumenta con la latencia $W$ (en segundos) y con la fracción de solicitudes rechazadas $r$ que percibe el usuario. Los valores fijos son $p = 0.002$ por día, $M = 20{,}000$, $U(0) = 100$, $\delta_0 = 0.003$ por día, $\kappa_W = 1.0$ por día por segundo, $\kappa_R = 0.5$ por día y $W_{\text{ok}} = 0.75$ segundos.

**Capa DES (clúster de servidores en la hora pico).** El clúster es una cola M/M/$c$/$K$. Las solicitudes llegan según un proceso de Poisson de tasa $\lambda = a\,U$ con $a = 0.0005$ solicitudes por segundo por usuario activo. Cada servidor atiende con tiempo exponencial de media 0.5 segundos, es decir $\mu = 2$ solicitudes por segundo. El sistema admite como máximo $K = 60$ solicitudes simultáneas, y si está lleno la solicitud nueva es rechazada. Cada vez que el SD necesita datos del DES, este simula 720 segundos de hora pico con la tasa y los servidores vigentes, descarta los primeros 120 segundos como calentamiento y devuelve la latencia promedio $W$ de las solicitudes aceptadas (tiempo en el sistema) y la fracción rechazada $r$.

**Política de autoescalado (evento de estado en el SD).** Sea $\rho_{\text{pred}}(t) = a\,U(t)/(c\,\mu)$ la utilización prevista. Cuando $\rho_{\text{pred}}$ cruza hacia arriba el umbral $\theta = 0.7$ y no hay un pedido pendiente, se ordenan 2 servidores que llegan $D$ días después, que es un evento de tiempo. Si al llegar $\rho_{\text{pred}}$ sigue por encima de $\theta$, se ordena otro pedido de inmediato. Al inicio, $c(0) = 4$ servidores.

**Acoplamiento.** Cada $H$ días el SD entrega al DES la tasa $\lambda = a\,U$ y el número de servidores $c$, y el DES devuelve $W$ y $r$, que el SD mantiene constantes hasta el siguiente intercambio. El valor base es $H = 1$ día.

**Parámetros de su grupo.** Sea $g$ el número de su grupo. Entonces:

$$q = 0.20 + 0.05\,((g - 1) \bmod 4), \qquad D = 3 + ((g - 1) \bmod 4) \text{ días}$$

Indique al inicio de su reporte su número de grupo y los valores de $q$ y $D$.

**Estructura de entregas.** La entrega parcial cubre los Tasks 1 y 2. La entrega final agrega el Task 3.

---

## Task 1 (Entrega Parcial)

### Task 1.1

Escriba sus predicciones antes de ejecutar cualquier simulación de esta hoja, y no las modifique después. Cada predicción debe incluir una dirección o un número y su razonamiento en dos o tres líneas.

a. Compare el híbrido con retroalimentación completa contra la cascada, que es el mismo modelo con $\kappa_W = \kappa_R = 0$, de modo que el DES no afecta a los usuarios. Prediga qué le ocurre a $U(100)$, a la latencia máxima y a la fracción máxima de solicitudes rechazadas al cerrar el lazo.

b. Prediga en qué día aproximado se emite el primer pedido de servidores y cuántos servidores habrá en el día 100.

c. Prediga qué cambia en los resultados al aumentar la ventana de intercambio de $H = 1$ a $H = 7$ días: qué métrica cambia, en qué dirección y por qué. Indique si espera que el resultado cambie de forma regular o irregular con $H$.

### Task 1.2

Deduzca a mano el equilibrio del lazo cuando la capacidad se mantiene fija, para saber hacia dónde tiende la plataforma antes de simularla. Use $c = 4$ y suponga que no hay rechazos.

a. Escriba la latencia de la fila M/M/$c$ en función de $U$ con las fórmulas de la sesión: la carga ofrecida $A = aU/\mu$, la probabilidad de esperar $C(c, A)$ de Erlang y $W = C(c, A)/(c\mu - aU) + 1/\mu$. Indique para qué valores de $U$ es válida.

b. Escriba la ecuación que debe cumplir el equilibrio $U^*$ del SD acoplado, igualando la adopción neta con el abandono evaluado con esa latencia. Resuélvala numéricamente y reporte $U^*$, la utilización $\rho$ y la latencia $W$ en ese equilibrio.

c. Explique en tres líneas por qué la plataforma no llega a $M$ con capacidad fija y qué papel cumple la retroalimentación. Compare con su predicción del Task 1.1a.

### Task 1.3

Pida a una IA que escriba el híbrido completo y entréguelo sin modificar. Este será su línea base, y en el Task 3.2 informará qué mejoró respecto de ella.

a. Entregue el modelo completo de esta hoja a la IA que prefiera, incluido un agente que escribe y ejecuta código, sin darle pistas sobre eventos de estado, conservación en la interfaz ni sensibilidad a la ventana. Guarde en un anexo el prompt exacto y el código tal como lo recibió.

b. Corra el código tal como lo recibió, una réplica con $H = 1$ y $c(0) = 4$ durante 100 días. Reporte $U(100)$, el número de servidores final, la latencia máxima y la fracción máxima de solicitudes rechazadas.

c. Copie literalmente las afirmaciones de la IA sobre la corrección del código, por ejemplo que los resultados son razonables o que la simulación coincide con la teoría, e indique en una línea qué verificó la IA de manera independiente y qué dio por supuesto.

---

## Task 2 (Entrega Parcial)

### Task 2.1

Verifique la capa DES por separado contra la teoría, sin el SD, con parámetros fijos.

a. Con $\lambda = 6$ solicitudes por segundo, $c = 4$ y $K$ muy grande, simule la fila con al menos 30 réplicas independientes de 720 segundos con calentamiento de 120. Compare la latencia, el tiempo de espera en cola y la utilización con las fórmulas de M/M/$c$, con intervalos de confianza del 95 %.

b. Con $\lambda = 10$, $c = 4$ y $K = 60$, la fila está sobrecargada. Compare la fracción rechazada simulada con la aproximación $1 - c\mu/\lambda$ y explique por qué es una aproximación y no un valor exacto.

c. Verifique con la ley de Little las tres cantidades obtenidas de su simulación y reporte el error relativo.

### Task 2.2

Verifique la capa SD por separado, con el abandono constante e igual a $\delta_0$.

a. Implemente su propio integrador de Runge-Kutta de cuarto orden y compare la trayectoria de $U(t)$ durante 100 días con la que entrega `solve_ivp` con tolerancia $10^{-10}$.

b. Muestre el error máximo contra el paso de integración para Euler y para Runge-Kutta de cuarto orden, en escala logarítmica, y estime el orden de cada método. Escoja el paso que usará en el resto de la hoja y justifíquelo.

### Task 2.3

Implemente la política de autoescalado como un evento de estado correctamente localizado y mida cuánto importa hacerlo bien.

a. Implemente el cruce de $\rho_{\text{pred}}$ con el umbral $\theta$ localizando el instante del evento dentro del paso, con `solve_ivp` y su opción de eventos o con bisección propia. Implemente el evento de tiempo de llegada de los servidores y la regla del reorden inmediato.

b. Implemente también una versión ingenua que revisa el umbral solo al final de cada paso de longitud $\Delta t$, con $\Delta t = 1$ día y con $\Delta t = 0.25$ días. Para cada versión reporte el día del primer pedido, el día en que llegan los servidores, la latencia máxima y $U(100)$.

c. Explique la diferencia entre las tres versiones con sus propios datos, y diga en qué condiciones de crecimiento y de retraso $D$ la detección tardía produce el mayor daño.

### Task 2.4

Verifique la interfaz entre las dos capas, que es donde los híbridos suelen fallar sin avisar.

a. En 30 ventanas del DES con $\lambda = 5$, compare el número de solicitudes generadas en la parte medida de la ventana con el valor esperado $\lambda$ por la duración. Verifique que el número medido tenga el comportamiento de un conteo de Poisson, por ejemplo comparando su media y su varianza.

b. En cada ventana compruebe la conservación de entidades: las solicitudes que llegaron deben ser iguales a las atendidas, más las rechazadas, más las que quedan dentro. Reporte el balance de al menos 10 ventanas, incluyendo ventanas con sobrecarga.

c. Describa en un párrafo la regla que usted escogió para el calentamiento de cada ventana y qué sesgo introduciría si la ventana comenzara con el sistema vacío y no se descartara nada.

---

## Task 3 (Entrega Final)

### Task 3.1

Mida cuánto depende el resultado del tamaño de la ventana de intercambio, y compárelo con su predicción del Task 1.1c.

a. Corra el híbrido completo con $H \in \{0.25, 0.5, 1, 2, 4, 7\}$ días, con al menos 20 réplicas independientes por valor y 100 días de horizonte. Para cada $H$ reporte, con intervalo de confianza del 95 %, $U(100)$, la latencia máxima y el error relativo cuadrático medio de $U(t)$ respecto a la referencia $H = 0.25$.

b. Grafique el error contra $H$ y describa su forma, regular o irregular. Si no es monótono, explique con su teoría qué lo causa, por ejemplo el ruido del DES o el retraso del lazo, y qué experimento adicional haría para distinguirlas.

c. Compare con su predicción del Task 1.1c y diga qué fue lo que no esperaba. Indique a partir de qué ventana no confiaría en el modelo, y con qué criterio.

### Task 3.2

Escriba el informe de delta entre su híbrido verificado y la línea base que le entregó la IA en el Task 1.3.

a. Presente una tabla con al menos cuatro diferencias concretas entre ambos modelos, por ejemplo la detección del evento de estado, la conservación en la interfaz, el integrador, el calentamiento de la ventana, la replicación y la verificación contra la teoría. Para cada una indique qué hacía la línea base, qué hace su modelo, la evidencia medida de que la mejora importa y cuánto cambió el resultado.

b. Indique al menos una cosa que la línea base de la IA hizo bien y que usted conservó, y al menos una que falló de una forma que no habría notado sin su verificación por capas.

c. Escriba un párrafo con la lección más general: qué debería pedirle a una IA, y qué debería verificar siempre usted, al construir un híbrido.

### Task 3.3

Con su híbrido verificado, evalúe políticas de autoescalado y recomiende una al director técnico.

a. Evalúe las doce políticas formadas por el umbral $\theta \in \{0.5, 0.6, 0.7, 0.8\}$ y el tamaño del pedido $k \in \{1, 2, 4\}$ servidores, con 20 réplicas por política y un horizonte de 150 días. Para cada política calcule el ingreso neto, definido como la suma diaria de los usuarios activos menos 40 veces la suma diaria de los servidores, y la probabilidad de que la fracción rechazada supere el 5 % en algún día.

b. Presente una tabla con las doce políticas, sus intervalos de confianza y los resultados, e indique cuáles políticas no se pueden distinguir estadísticamente entre sí.

c. Recomiende una política. Justifique por qué la escogió frente a las que no se distinguen de ella, y qué consideraría el director técnico que no está en la tabla.

### Task 3.4

Prepare la defensa de su recomendación ante el director técnico, quien depende de ella para pedir presupuesto. Responda con honestidad sobre los límites de lo que usted construyó.

a. Escriba las tres preguntas más difíciles que el director podría hacer sobre los límites del modelo y su respuesta a cada una. Las respuestas deben apoyarse en lo que descubrieron en los Tasks 1 a 3, por ejemplo la sensibilidad a la ventana, los errores de la línea base o la dependencia de los coeficientes de abandono que se supusieron.

b. Indique qué dato real pediría a la empresa antes de que el director firme la decisión, cómo lo usaría para ajustar el modelo y qué parte de su recomendación cambiaría si el dato resulta distinto de lo supuesto.

c. Escriba una frase final con las condiciones bajo las cuales retiraría su recomendación.

---

## Entregas en Canvas

1. Documento PDF con las respuestas a cada task.
2. En la entrega parcial se espera que entreguen lo **señalado**. En la entrega final deben entregar **TODOS LOS TASK**.
3. Archivo .ipynb, o link a repositorio de GitHub (**No se acepta entregas en otros medios**).
   - a. El código debe estar comentado explicando la relación con las fórmulas de las diapositivas.
4. Anexo con el prompt y el código original de la línea base generado por la IA (Task 1.3).
5. Las predicciones del Task 1.1 deben entregarse tal como las escribió antes de simular, sin modificaciones posteriores.

## Evaluación

1. [1.20 pt] Task 1
2. [1.20 pt] Task 2
3. [1.60 pt] Task 3

**Total: 4.0 pts**
