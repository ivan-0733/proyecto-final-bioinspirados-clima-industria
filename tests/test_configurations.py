#!/usr/bin/env python
"""
Test de Configuraciones - Validación Rápida
============================================

Prueba diferentes configuraciones para asegurar que todas funcionan correctamente.
Usa parámetros reducidos (10 generaciones, 20 población) para tests rápidos.
"""

import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import Orchestrator


def create_test_config(
    scenario: str,
    mutation_method: str,
    decomposition_method: str = "tchebycheff",
    population_size: int = 20,
    generations: int = 10,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.4,
) -> dict:
    """
    Crea configuración de test basada en parámetros.
    
    Args:
        scenario: "scenario_1" o "scenario_2"
        mutation_method: "fallback", "conservative", "template", "guided", "mixed"
        decomposition_method: "tchebycheff", "weighted_sum", "pbi"
        population_size: Tamaño de población (default: 20 para tests rápidos)
        generations: Número de generaciones (default: 10 para tests rápidos)
        crossover_prob: Probabilidad inicial de crossover
        mutation_prob: Probabilidad inicial de mutación
    
    Returns:
        Dict con configuración completa
    """
    base_config = {
        "experiment": {
            "name": f"test_{mutation_method}_{scenario}",
            "scenario": scenario,
            "description": f"Test rápido de {mutation_method} en {scenario}",
            "random_seed": 42,
            "output_root": "results"
        },
        "use_sampling": True,
        "dataset": {
            "use_sample": False,
            "sample_fraction": 1.0
        },
        "objectives": {
            "selected": (
                ["casual-supp", "casual-conf", "maxConf"] 
                if scenario == "scenario_1" 
                else ["jaccard", "cosine", "phi", "kappa"]
            )
        },
        "algorithm": {
            "population_size": population_size,
            "generations": generations,
            "logging_interval": 5,
            "termination": {
                "enabled": True,
                "ftol": 0.0001,
                "period": 5
            },
            "initialization": {
                "max_attempts": 1000
            },
            "decomposition": {
                "method": decomposition_method,
                "params": {}
            },
            "neighborhood": {
                "size": 3,
                "replacement_size": 3,
                "selection_probability": 0.3
            },
            "operators": {
                "crossover": {
                    "n_points": 4,
                    "probability": {
                        "initial": crossover_prob,
                        "min": 0.5,
                        "max": 0.8
                    }
                },
                "mutation": {
                    "method": mutation_method,
                    "active_ops": ["extension", "contraction", "replacement"],
                    "probability": {
                        "initial": mutation_prob,
                        "min": 0.3,
                        "max": 0.6
                    },
                    "repair_attempts": 5,
                    "duplicate_attempts": 5,
                    "timeout": 2.0,
                    "timeout_per_attempt": 10.0
                }
            },
            "stuck_detection": {
                "enabled": True,
                "window": 3,
                "min_new": 0,
                "hv_window": 5,
                "hv_tol": 0.0001
            }
        },
        "constraints": {
            "rule_validity": {
                "min_antecedent_items": 1,
                "min_consequent_items": 1,
                "max_antecedent_items": 4,
                "max_consequent_items": 2
            },
            "metric_thresholds": {
                "casual-supp": {"min": 0.0, "max": 2.0},
                "casual-conf": {"min": 0.0, "max": 1.0},
                "maxConf": {"min": 0.0, "max": 1.0},
                "jaccard": {"min": 0.0, "max": 1.0},
                "cosine": {"min": 0.0, "max": 1.0},
                "phi": {"min": -1.0, "max": 1.0},
                "kappa": {"min": -1.0, "max": 1.0}
            },
            "exclusions": {
                "fixed_consequents": [],
                "forbidden_pairs": []
            }
        },
        "termination": {
            "enabled": True,
            "ftol": 0.0001,
            "period": 5
        },
        "initialization": {
            "use_pregenerated": True,
            "max_attempts": 1000
        },
        "outputs": {
            "save_populations": True,
            "save_pareto_fronts": True,
            "generate_plots": False,  # Deshabilitado para tests rápidos
            "log_discarded_rules": True
        }
    }
    
    return base_config


def run_test(config_name: str, config: dict, verbose: bool = True) -> dict:
    """
    Ejecuta un test con la configuración dada.
    
    Args:
        config_name: Nombre descriptivo del test
        config: Diccionario de configuración
        verbose: Si mostrar detalles durante ejecución
    
    Returns:
        Dict con resultados del test
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"TEST: {config_name}")
        print(f"{'='*60}")
        print(f"Escenario: {config['experiment']['scenario']}")
        print(f"Mutación: {config['algorithm']['operators']['mutation']['method']}")
        print(f"Población: {config['algorithm']['population_size']}")
        print(f"Generaciones: {config['algorithm']['generations']}")
    
    # Crear archivo temporal de configuración
    config_path = Path(__file__).parent / f"temp_config_{config_name}.json"
    
    try:
        # Guardar configuración
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # Ejecutar optimización
        start_time = time.time()
        orchestrator = Orchestrator(str(config_path))
        result = orchestrator.run()
        elapsed_time = time.time() - start_time
        
        # Analizar resultados
        experiment_dir = getattr(result, 'experiment_dir', None)
        if experiment_dir:
            final_pareto_path = Path(experiment_dir) / "final_pareto_historical.csv"
            
            if final_pareto_path.exists():
                import pandas as pd
                df = pd.read_csv(final_pareto_path)
                unique_solutions = len(df)
            else:
                unique_solutions = 0
        else:
            unique_solutions = 0
        
        test_result = {
            "name": config_name,
            "status": "✓ PASSED",
            "time": elapsed_time,
            "unique_solutions": unique_solutions,
            "generations_completed": getattr(result, 'generations_completed', 'unknown'),
            "experiment_dir": experiment_dir or "N/A"
        }
        
        if verbose:
            print(f"\n✓ Test completado en {elapsed_time:.2f}s")
            print(f"  Soluciones únicas: {unique_solutions}")
            print(f"  Generaciones completadas: {test_result['generations_completed']}")
        
        return test_result
        
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        test_result = {
            "name": config_name,
            "status": "✗ FAILED",
            "error": str(e),
            "time": elapsed
        }
        
        if verbose:
            print(f"\n✗ Test falló: {e}")
        
        return test_result
        
    finally:
        # Limpiar archivo temporal
        if config_path.exists():
            config_path.unlink()


def main():
    """
    Ejecuta batería completa de tests de configuración.
    """
    print("="*70)
    print("TEST DE CONFIGURACIONES - Validación Exhaustiva")
    print("="*70)
    print("\nProbando diferentes configuraciones con parámetros reducidos")
    print("(10 generaciones, 20 población) para validación rápida")
    print("\n4 PARTES:")
    print("  1. Métodos de Descomposición (tchebycheff, weighted_sum, pbi)")
    print("  2. Estrategias de Mutación (fallback, conservative, template, guided)")
    print("  3. Combinaciones Específicas (decomposition + mutation)")
    print("  4. Variaciones de Hiperparámetros (población, crossover, mutation)")
    print()
    
    # Definir tests a ejecutar
    test_configs = []
    
    # ========== PARTE 1: Tests de Métodos de Descomposición ==========
    print("\n[PARTE 1] Testing Métodos de Descomposición")
    print("Probando: tchebycheff, weighted_sum, pbi con mutation conservative\n")
    
    for decomp in ["tchebycheff", "weighted_sum", "pbi"]:
        test_configs.extend([
            {
                "name": f"S1_{decomp}_Conservative",
                "config": create_test_config(
                    scenario="scenario_1",
                    mutation_method="conservative",
                    decomposition_method=decomp,
                    crossover_prob=0.7,
                    mutation_prob=0.4
                )
            },
            {
                "name": f"S2_{decomp}_Guided",
                "config": create_test_config(
                    scenario="scenario_2",
                    mutation_method="guided",
                    decomposition_method=decomp,
                    crossover_prob=0.7,
                    mutation_prob=0.4
                )
            },
        ])
    
    # ========== PARTE 2: Tests de Estrategias de Mutación ==========
    print("\n[PARTE 2] Testing Estrategias de Mutación (con tchebycheff)")
    print("Probando: fallback, conservative, template, guided\n")
    
    for mutation in ["fallback", "conservative", "template", "guided"]:
        test_configs.extend([
            {
                "name": f"S1_Tcheby_{mutation}",
                "config": create_test_config(
                    scenario="scenario_1",
                    mutation_method=mutation,
                    decomposition_method="tchebycheff",
                    crossover_prob=0.7,
                    mutation_prob=0.4
                )
            },
            {
                "name": f"S2_Tcheby_{mutation}",
                "config": create_test_config(
                    scenario="scenario_2",
                    mutation_method=mutation,
                    decomposition_method="tchebycheff",
                    crossover_prob=0.7,
                    mutation_prob=0.4
                )
            },
        ])
    
    # ========== PARTE 3: Combinaciones Interesantes ==========
    print("\n[PARTE 3] Testing Combinaciones Específicas")
    print("Probando combinaciones de decomposition + mutation óptimas\n")
    
    test_configs.extend([
        # Weighted_sum + Fallback (diversidad)
        {
            "name": "S1_WeightedSum_Fallback",
            "config": create_test_config(
                scenario="scenario_1",
                mutation_method="fallback",
                decomposition_method="weighted_sum",
                crossover_prob=0.7,
                mutation_prob=0.4
            )
        },
        # PBI + Template (rapidez)
        {
            "name": "S1_PBI_Template",
            "config": create_test_config(
                scenario="scenario_1",
                mutation_method="template",
                decomposition_method="pbi",
                crossover_prob=0.7,
                mutation_prob=0.5
            )
        },
        # Weighted_sum + Guided (consistencia)
        {
            "name": "S2_WeightedSum_Guided",
            "config": create_test_config(
                scenario="scenario_2",
                mutation_method="guided",
                decomposition_method="weighted_sum",
                crossover_prob=0.7,
                mutation_prob=0.4
            )
        },
        # PBI + Conservative (balance)
        {
            "name": "S2_PBI_Conservative",
            "config": create_test_config(
                scenario="scenario_2",
                mutation_method="conservative",
                decomposition_method="pbi",
                crossover_prob=0.7,
                mutation_prob=0.4
            )
        },
    ])
    
    # ========== PARTE 4: Variaciones de Hiperparámetros ==========
    print("\n[PARTE 4] Testing Variaciones de Hiperparámetros")
    print("Probando diferentes poblaciones y probabilidades\n")
    
    test_configs.extend([
        {
            "name": "S1_SmallPop_HighCrossover",
            "config": create_test_config(
                scenario="scenario_1",
                mutation_method="conservative",
                decomposition_method="tchebycheff",
                population_size=15,
                crossover_prob=0.8,
                mutation_prob=0.3
            )
        },
        {
            "name": "S1_LargePop_LowCrossover",
            "config": create_test_config(
                scenario="scenario_1",
                mutation_method="conservative",
                decomposition_method="tchebycheff",
                population_size=30,
                crossover_prob=0.5,
                mutation_prob=0.6
            )
        },
    ])
    
    # Ejecutar todos los tests
    results = []
    start_time = time.time()
    
    for i, test_spec in enumerate(test_configs, 1):
        print(f"\n[{i}/{len(test_configs)}] Ejecutando: {test_spec['name']}")
        result = run_test(test_spec["name"], test_spec["config"], verbose=False)
        results.append(result)
        
        # Mostrar progreso compacto
        if result["status"] == "✓ PASSED":
            print(f"  ✓ {result['time']:.1f}s | {result['unique_solutions']} soluciones | Gen: {result['generations_completed']}")
        else:
            print(f"  ✗ FAILED: {result.get('error', 'Unknown error')}")
    
    total_time = time.time() - start_time
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE TESTS")
    print("="*70)
    
    passed = sum(1 for r in results if r["status"] == "✓ PASSED")
    failed = len(results) - passed
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Pasados: {passed} ✓")
    print(f"Fallidos: {failed} ✗")
    print(f"Tiempo total: {total_time:.2f}s")
    print(f"Tiempo promedio: {total_time/len(results):.2f}s por test")
    
    # Agrupar por tipo de test
    decomp_tests = [r for r in results if any(d in r['name'] for d in ['tchebycheff', 'weighted_sum', 'pbi']) and 'Tcheby' not in r['name']]
    mutation_tests = [r for r in results if 'Tcheby_' in r['name']]
    combo_tests = [r for r in results if any(c in r['name'] for c in ['WeightedSum_', 'PBI_']) and 'Tcheby' not in r['name']]
    hyperparam_tests = [r for r in results if 'Pop' in r['name']]
    
    print("\nResultados por categoría:")
    print(f"  Descomposición: {sum(1 for r in decomp_tests if r['status']=='✓ PASSED')}/{len(decomp_tests)} ✓")
    print(f"  Mutación: {sum(1 for r in mutation_tests if r['status']=='✓ PASSED')}/{len(mutation_tests)} ✓")
    print(f"  Combinaciones: {sum(1 for r in combo_tests if r['status']=='✓ PASSED')}/{len(combo_tests)} ✓")
    print(f"  Hiperparámetros: {sum(1 for r in hyperparam_tests if r['status']=='✓ PASSED')}/{len(hyperparam_tests)} ✓")
    
    # Tabla de resultados
    print("\nResultados detallados:")
    print(f"{'Test':<40} {'Status':<12} {'Tiempo':<10} {'Soluciones':<12}")
    print("-"*75)
    
    for result in results:
        if result["status"] == "✓ PASSED":
            print(f"{result['name']:<40} {result['status']:<12} {result['time']:<10.2f} {result['unique_solutions']:<12}")
        else:
            error_preview = result.get('error', 'Unknown')[:30]
            print(f"{result['name']:<40} {result['status']:<12} {result.get('time', 0):<10.2f} {'N/A':<12}")
    
    # Análisis de rendimiento
    if passed > 0:
        print("\n" + "="*70)
        print("ANÁLISIS DE RENDIMIENTO")
        print("="*70)
        
        successful_results = [r for r in results if r["status"] == "✓ PASSED"]
        
        fastest = min(successful_results, key=lambda x: x["time"])
        slowest = max(successful_results, key=lambda x: x["time"])
        most_diverse = max(successful_results, key=lambda x: x["unique_solutions"])
        
        print(f"\nMás rápido: {fastest['name']} ({fastest['time']:.2f}s)")
        print(f"Más lento: {slowest['name']} ({slowest['time']:.2f}s)")
        print(f"Más diverso: {most_diverse['name']} ({most_diverse['unique_solutions']} soluciones)")
        
        avg_time = sum(r["time"] for r in successful_results) / len(successful_results)
        avg_diversity = sum(r["unique_solutions"] for r in successful_results) / len(successful_results)
        
        print(f"\nPromedios:")
        print(f"  Tiempo: {avg_time:.2f}s")
        print(f"  Diversidad: {avg_diversity:.1f} soluciones")
        
        # Análisis por método de descomposición
        if decomp_tests:
            print("\n" + "-"*70)
            print("Análisis por Método de Descomposición:")
            for decomp_method in ["tchebycheff", "weighted_sum", "pbi"]:
                decomp_results = [r for r in successful_results if decomp_method in r['name'].lower()]
                if decomp_results:
                    avg_div = sum(r['unique_solutions'] for r in decomp_results) / len(decomp_results)
                    avg_t = sum(r['time'] for r in decomp_results) / len(decomp_results)
                    print(f"  {decomp_method:15s}: {len(decomp_results)} tests, Avg div={avg_div:.1f}, Avg time={avg_t:.2f}s")
        
        # Análisis por estrategia de mutación
        if mutation_tests:
            print("\n" + "-"*70)
            print("Análisis por Estrategia de Mutación:")
            for mut_method in ["fallback", "conservative", "template", "guided"]:
                mut_results = [r for r in successful_results if mut_method in r['name'].lower()]
                if mut_results:
                    avg_div = sum(r['unique_solutions'] for r in mut_results) / len(mut_results)
                    avg_t = sum(r['time'] for r in mut_results) / len(mut_results)
                    print(f"  {mut_method:15s}: {len(mut_results)} tests, Avg div={avg_div:.1f}, Avg time={avg_t:.2f}s")
    
    # Verificar si todos pasaron
    if failed == 0:
        print("\n" + "="*70)
        print("✓ TODAS LAS CONFIGURACIONES VALIDADAS CORRECTAMENTE")
        print("="*70)
        print("\n✨ Configuraciones probadas:")
        print("  • 3 métodos de descomposición: tchebycheff, weighted_sum, pbi")
        print("  • 4 estrategias de mutación: fallback, conservative, template, guided")
        print("  • Múltiples combinaciones y variaciones de hiperparámetros")
        return 0
    else:
        print("\n" + "="*70)
        print(f"✗ {failed} CONFIGURACIONES FALLARON")
        print("="*70)
        
        # Mostrar errores
        failed_results = [r for r in results if r["status"] == "✗ FAILED"]
        if failed_results:
            print("\nErrores encontrados:")
            for result in failed_results[:5]:  # Mostrar solo los primeros 5
                print(f"  • {result['name']}: {result.get('error', 'Unknown error')[:80]}")
            if len(failed_results) > 5:
                print(f"  ... y {len(failed_results)-5} más")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
