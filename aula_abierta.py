#!/usr/bin/env python3
"""Aula Abierta: SD continuo + DES M/M/c/K. Solo biblioteca estándar.

Ejecutar: python aula_abierta.py
Reproducir la réplica entregada: Python 3, semilla 42, H=1, horizonte=100.

Convenciones:
* Cada DES comienza vacío y es una ventana independiente de hora pico.
* Se mide la cohorte de llegadas en [120,720) segundos; las aceptadas
  se siguen hasta su salida, aunque esta ocurra después de 720 s.
* K incluye solicitudes en servicio y en espera; disciplina FCFS.
* Intercambios en t=0,H,2H,... estrictamente anteriores al horizonte.
  No se genera una nueva observación DES al terminar el día 100.
* W y r se mantienen constantes entre intercambios, incluso si llega
  un pedido. Una entrega cambia c inmediatamente y afecta al DES siguiente.
* Si coinciden entrega e intercambio, se procesa primero la entrega.
* Los cruces de utilización se detectan en tiempo continuo; los pedidos
  llegan exactamente D días después. Se admite un único pedido pendiente.
* Los máximos reportados son máximos de los promedios por ventana DES,
  no el máximo tiempo individual de una solicitud.
"""

import argparse
from collections import deque
from dataclasses import dataclass, asdict
import heapq
import json
import math
import random


@dataclass(frozen=True)
class Parameters:
    p: float = 0.002
    q: float = 0.30
    M: float = 20000.0
    U0: float = 100.0
    delta0: float = 0.003
    kappa_W: float = 1.0
    kappa_R: float = 0.5
    Wok: float = 0.75
    a: float = 0.0005
    mu: float = 2.0
    K: int = 60
    theta: float = 0.7
    batch: int = 2
    delay: float = 5.0
    peak: float = 720.0
    warmup: float = 120.0


def des(arrival_rate, servers, rng, p):
    """Simulación por eventos: reloj en segundos, cola FCFS finita."""
    assert arrival_rate >= 0 and servers >= 1 and p.K >= 1
    assert 0 <= p.warmup < p.peak
    if arrival_rate == 0:
        return dict(W=1 / p.mu, r=0.0, arrivals=0, rejected=0,
                    accepted=0, latency_count=0)
    # Heap de salidas: (tiempo de salida, id, llegada, pertenece a cohorte).
    busy = []
    waiting = deque()
    next_arrival = rng.expovariate(arrival_rate)
    arrivals = rejected = accepted = completed = serial = 0
    latency_sum = 0.0
    t = 0.0

    def start_service(arrived, measured, now):
        nonlocal serial
        serial += 1
        heapq.heappush(busy, (now + rng.expovariate(p.mu), serial,
                              arrived, measured))

    while next_arrival < p.peak or busy:
        next_departure = busy[0][0] if busy else math.inf
        # No nuevas llegadas tras 720 s; se drena la cohorte pendiente.
        candidate_arrival = next_arrival if next_arrival < p.peak else math.inf
        if candidate_arrival < next_departure:
            t = candidate_arrival
            measured = t >= p.warmup
            if measured:
                arrivals += 1
            if len(busy) + len(waiting) >= p.K:
                if measured:
                    rejected += 1
            else:
                if measured:
                    accepted += 1
                if len(busy) < servers:
                    start_service(t, measured, t)
                else:
                    waiting.append((t, measured))
            next_arrival = t + rng.expovariate(arrival_rate)
        else:
            t, _, arrived, measured = heapq.heappop(busy)
            if measured:
                completed += 1
                latency_sum += t - arrived
            if waiting:
                arrived, measured = waiting.popleft()
                start_service(arrived, measured, t)
        assert len(busy) <= servers
        assert len(busy) + len(waiting) <= p.K
    assert completed == accepted and arrivals == accepted + rejected
    # Ventanas sin aceptadas: media de servicio como valor de respaldo explícito.
    return dict(W=latency_sum / completed if completed else 1 / p.mu,
                r=rejected / arrivals if arrivals else 0.0,
                arrivals=arrivals, rejected=rejected, accepted=accepted,
                latency_count=completed)


def sd_flow(U, days, delta, p):
    """Solución exacta de U'=A+B*U-C*U² con delta constante.

    A=p*M, B=q-p-delta, C=q/M; raíces up>0 y um<0.
    (U(t)-up)/(U(t)-um) = (U(0)-up)/(U(0)-um)*exp(-s*t).
    """
    A, B, C = p.p * p.M, p.q - p.p - delta, p.q / p.M
    s = math.sqrt(B * B + 4 * A * C)
    # Cálculo estable de ambas raíces, incluso con abandono alto.
    if B >= 0:
        up = (B + s) / (2 * C)
        um = -A / (C * up)
    else:
        um = (B - s) / (2 * C)
        up = -A / (C * um)
    z = (U - up) / (U - um) * math.exp(-s * days)
    return (up - z * um) / (1 - z)


def upward_crossing(U, duration, delta, target, p):
    """Primer cruce ascendente dentro del segmento; bisección de flujo exacto."""
    if not (U < target <= sd_flow(U, duration, delta, p)):
        return None
    lo, hi = 0.0, duration
    for _ in range(60):
        middle = (lo + hi) / 2
        if sd_flow(U, middle, delta, p) < target:
            lo = middle
        else:
            hi = middle
    return hi


def simulate(H=1.0, horizon=100.0, initial_servers=4, seed=42, p=None):
    """Una réplica híbrida; tiempo externo en días, DES interno en segundos."""
    p = p or Parameters()
    if H <= 0 or horizon <= 0 or initial_servers < 1:
        raise ValueError('Se requieren H>0, horizonte>0 y servidores>=1.')
    rng = random.Random(seed)
    t, U, c = 0.0, p.U0, initial_servers
    W, r = 1 / p.mu, 0.0
    due = None
    exchange_index = 0
    exchanges, events = [], []
    eps = 1e-10

    def order(reason):
        nonlocal due
        assert due is None
        due = t + p.delay
        events.append(dict(type='pedido', day=t, U=U, c=c,
                           rho=p.a * U / (c * p.mu), due=due,
                           reason=reason))

    while True:
        # Se procesan también entregas exactamente en el horizonte.
        if due is not None and due <= t + eps:
            c += p.batch
            due = None
            events.append(dict(type='entrega', day=t, U=U, c=c,
                               rho=p.a * U / (c * p.mu)))
            if p.a * U / (c * p.mu) > p.theta:
                order('utilizacion superior al umbral despues de entrega')
        if t >= horizon - eps:
            break

        next_exchange = exchange_index * H
        if next_exchange <= t + eps:
            metrics = des(p.a * U, c, rng, p)
            W, r = metrics['W'], metrics['r']
            exchanges.append(dict(day=t, U=U, c=c, lambda_s=p.a * U,
                                  rho=p.a * U / (c * p.mu), **metrics))
            exchange_index += 1
            next_exchange = exchange_index * H

        # Permite también inicializaciones con rho ya superior al umbral.
        if due is None and p.a * U / (c * p.mu) > p.theta:
            order('utilizacion inicial superior al umbral')
        delta = (p.delta0 + p.kappa_W * max(0.0, W - p.Wok)
                 + p.kappa_R * r)
        boundary = min(horizon, next_exchange,
                       due if due is not None else math.inf)
        duration = boundary - t
        assert duration > 0
        crossing = None
        if due is None:
            threshold_users = p.theta * c * p.mu / p.a
            crossing = upward_crossing(U, duration, delta, threshold_users, p)
        if crossing is not None:
            U = sd_flow(U, crossing, delta, p)
            t += crossing
            order('cruce ascendente')
        else:
            U = sd_flow(U, duration, delta, p)
            t = boundary
        assert 0 <= U <= p.M

    max_W = max(exchanges, key=lambda row: row['W'])
    max_r = max(exchanges, key=lambda row: row['r'])
    summary = dict(seed=seed, H_days=H, horizon_days=horizon,
                   initial_servers=initial_servers, U_final=U,
                   final_servers=c, W_max_s=max_W['W'],
                   W_max_day=max_W['day'], r_max=max_r['r'],
                   r_max_day=max_r['day'],
                   orders=sum(e['type'] == 'pedido' for e in events),
                   deliveries=sum(e['type'] == 'entrega' for e in events),
                   pending_delivery_day=due, DES_windows=len(exchanges))
    return dict(parameters=asdict(p), summary=summary,
                events=events, exchanges=exchanges)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--H', type=float, default=1.0)
    parser.add_argument('--days', type=float, default=100.0)
    parser.add_argument('--servers', type=int, default=4)
    parser.add_argument('--full', action='store_true',
                        help='Imprime también parámetros y las ventanas DES.')
    args = parser.parse_args()
    result = simulate(args.H, args.days, args.servers, args.seed)
    output = result if args.full else dict(summary=result['summary'],
                                           events=result['events'])
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
