# src/observer_operator.py
# -----------------------------------------------------------
# LomonosovSaturatedInfinity — Algorithm of the Saturated Infinity
# Автор: Владимир Гончаров, 2025
#
# Идея:
#   Унитарный оператор с глобальной фазой e^{iφ} действует на состояние наблюдателя |ψ0⟩
#   так, что изменяется только фаза, но не наблюдаемые величины. Это моделирует
#   "неподвижного наблюдателя" в "бесконечном многообразии" — присутствие без перемещения.
#
# Назначение файла:
#   Демонстрационный, медитативный код. Не для численных расчётов в классическом смысле,
#   а для фиксации идеи и проверки инвариантности нормы/скалярного произведения.
# -----------------------------------------------------------

from __future__ import annotations
import numpy as np

def apply_unitary_with_global_phase(psi0: np.ndarray, phi: float) -> np.ndarray:
    """
    Применяет унитарный оператор вида U = e^{i*phi} * I к начальному состоянию psi0.
    Возвращает новое состояние psi_t.
    """
    identity = np.eye(len(psi0), dtype=complex)
    U = np.exp(1j * phi) * identity
    return U @ psi0

def is_norm_invariant(psi_a: np.ndarray, psi_b: np.ndarray, atol: float = 1e-12) -> bool:
    """Проверяет, что нормы состояний совпадают (инвариантность нормы при унитарном действии)."""
    return np.allclose(np.vdot(psi_a, psi_a), np.vdot(psi_b, psi_b), atol=atol)

def differs_by_global_phase(psi_a: np.ndarray, psi_b: np.ndarray, phi: float, atol: float = 1e-12) -> bool:
    """
    Проверяет, что psi_b = e^{i*phi} * psi_a (т.е. состояния различаются лишь глобальной фазой).
    """
    return np.allclose(psi_b, np.exp(1j * phi) * psi_a, atol=atol)

def main() -> None:
    # Начальное состояние наблюдателя |ψ0⟩
    psi_0 = np.array([1+0j, 0+0j], dtype=complex)

    # Фиксированная фаза φ (можно менять)
    phi = np.pi / 7  # "дыхание" восприятия, не меняющее наблюдаемых величин

    # Применяем унитарный оператор U = e^{iφ} I
    psi_t = apply_unitary_with_global_phase(psi_0, phi)

    # Диагностика инвариантности
    norm_invariant = is_norm_invariant(psi_0, psi_t)
    only_phase = differs_by_global_phase(psi_0, psi_t, phi)

    print("Исходное состояние |ψ0⟩:", psi_0)
    print("После эволюции U(t) |ψ0⟩:", psi_t)
    print("Норма сохраняется :", norm_invariant)
    print("Различие лишь фаза:", only_phase)
    print("Фаза φ (рад)      :", phi)
    # Покажем относительную фазу первой ненулевой компоненты (если есть)
    idx = int(np.argmax(np.abs(psi_0) > 0))
    rel_phase = np.angle(psi_t[idx] / psi_0[idx]) if psi_0[idx] != 0 else 0.0
    print("Измеренная относительная фаза компоненты:", rel_phase)

if __name__ == "__main__":
    main()
