# src/observer_operator.py
# ----------------------------------------------------------------------------------------
# LomonosovSaturatedInfinity — Algorithm of the Saturated Infinity.
# Автор: Владимир Гончаров, 2025
#
# Идея:
#   Унитарный оператор с глобальной фазой e^{iφ} действует на состояние наблюдателя |ψ0⟩
#   так, что изменяется только фаза, но не наблюдаемые величины. Это моделирует
#   «неподвижного наблюдателя» в «бесконечном многообразии» — присутствие без перемещения.
#
# Назначение файла:
#   Демонстрационный, медитативный код. Не для численных расчётов в классическом смысле,
#   а для фиксации идеи и проверки инвариантности нормы/скалярного произведения.
# ----------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

ArrayC = np.ndarray


@dataclass(frozen=True)
class EvolutionResult:
    psi_0: ArrayC
    psi_t: ArrayC
    U: ArrayC
    phi: float
    norm_invariant: bool
    only_global_phase: bool
    rel_phase_component0: float


def normalize(psi: ArrayC, eps: float = 1e-12) -> ArrayC:
    """Нормирует вектор состояния до единичной нормы."""
    nrm = np.linalg.norm(psi)
    if nrm < eps:
        raise ValueError("Норма вектора состояния слишком мала.")
    return psi / nrm


def unitary_global_phase(dim: int, phi: float) -> ArrayC:
    """Унитарный оператор вида U = e^{iφ} · I_dim."""
    return np.exp(1j * phi) * np.eye(dim, dtype=complex)


def is_unitary(U: ArrayC, eps: float = 1e-10) -> bool:
    """Проверка унитарности: U†U ≈ I."""
    I = np.eye(U.shape[0], dtype=complex)
    return np.allclose(U.conj().T @ U, I, atol=eps)


def norm_preserved(psi_0: ArrayC, psi_t: ArrayC, eps: float = 1e-10) -> bool:
    """Инвариантность нормы: ||psi_t|| ≈ ||psi_0||."""
    return np.allclose(np.linalg.norm(psi_t), np.linalg.norm(psi_0), atol=eps)


def differs_by_global_phase(psi_0: ArrayC, psi_t: ArrayC, eps: float = 1e-10) -> bool:
    """
    Проверяет, что psi_t = e^{iφ} · psi_0 для некоторого φ.
    Если psi_0 содержит нули, берём ненулевые компоненты.
    """
    nz = np.where(np.abs(psi_0) > eps)[0]
    if len(nz) == 0:
        # Тривиальный случай: |0⟩ → |0⟩
        return True
        # Оценим фазу по первой ненулевой компоненте   
    k = nz[0]
    ratio = psi_t[k] / psi_0[k]
    if np.abs(ratio) < eps:
        return False
        # Нормализуем обе волновые функции и сравним на равенство «с точностью до фазы»
    phi = np.angle(ratio)
    phase = np.exp(1j * phi)
    return np.allclose(psi_t, phase * psi_0, atol=1e-9)


def evolve_with_global_phase(psi_0: ArrayC, phi: float, eps: float = 1e-12) -> EvolutionResult:
    """Эволюция состояния под действием глобальной фазы e^{iφ} I."""
    psi_0 = np.asarray(psi_0, dtype=complex)
    if psi_0.ndim != 1:
        raise ValueError("Ожидается одномерный вектор состояния.")
    psi_0 = normalize(psi_0, eps=eps)

    dim = psi_0.shape[0]
    U = unitary_global_phase(dim, phi)
    assert is_unitary(U), "Сконструированный оператор должен быть унитарным."

    psi_t = U @ psi_0
    norm_ok = norm_preserved(psi_0, psi_t)
    only_phase = differs_by_global_phase(psi_0, psi_t)

    # Относительная фаза первой ненулевой компоненты
    nz = np.where(np.abs(psi_0) > 1e-12)[0]
    rel_phase = float(np.angle(psi_t[nz[0]] / psi_0[nz[0]])) if len(nz) else 0.0

    return EvolutionResult(
        psi_0=psi_0,
        psi_t=psi_t,
        U=U,
        phi=float(phi),
        norm_invariant=bool(norm_ok),
        only_global_phase=bool(only_phase),
        rel_phase_component0=rel_phase,
    )


def main() -> None:
    # Базовый пример: двухмерное состояние
    psi_0 = np.array([1 + 0j, 0 + 0j])  # |0⟩
    phi = np.pi / 7                     # глобальная фаза

    result = evolve_with_global_phase(psi_0, phi)

    print("Исходное состояние |ψ0⟩:", result.psi_0)
    print("После эволюции U(t)|ψ0⟩:", result.psi_t)
    print("U унитарен        :", is_unitary(result.U))
    print("Норма сохраняется :", result.norm_invariant)
    print("Различие лишь фаза:", result.only_global_phase)
    print("Фаза φ (рад)      :", result.phi)
    print("Относительная фаза:", result.rel_phase_component0)


if __name__ == "__main__":
    main()
