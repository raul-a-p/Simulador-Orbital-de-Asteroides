# 🌌 Simulador Orbital de Asteroides

Simulador físico de trajetórias orbitais de asteroides usando o método numérico Runge-Kutta de 4ª ordem (RK4) para resolver o problema gravitacional de N-corpos.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Required-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Required-green.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Sobre o Projeto

Este simulador foi desenvolvido como projeto de Computação Científica na disciplina F 625 do IFGW e permite:

- ✨ Simular trajetórias orbitais de asteroides no Sistema Solar
- 🎯 Detectar colisões e calcular parâmetros de impacto
- 🌍 Incluir múltiplos corpos celestes (Sol, planetas, Lua)
- 📊 Análise estatística via simulação Monte Carlo
- 🎬 Animações interativas das órbitas
- 📈 Validação física (conservação de energia e momento angular)


## 💻 Uso

Versões do simulador:
- Versão Python: instale o "simulador_orbital_asteroides_v2" e confira instruções de uso em "exemplos_v2" (recomendado)
- Versão Web: [Simulador de Asteroides](https://orbitalapp-tte5ngjs.manus.space/) (em desenvolvimento)


## 📊 Funcionalidades

### Cenários Pré-configurados
- 🌍 **Terra-Sol**: Validação do integrador
- ☄️ **Apophis**: Asteroide real (aproximação em 2029)
- 💥 **Impacto**: Colisão entre Terra e asteroide
- 🌙 **Terra-Lua**: Sistema com Lua e asteroide customizável
- 🪐 **Sistema Solar**: 8 planetas + cometa interestelar

### Análises Disponíveis
- Trajetórias orbitais 2D (estáticas e animadas)
- Distância temporal entre corpos
- Conservação de energia
- Detecção de colisões
- Parâmetros de impacto (energia, TNT equivalente, cratera)
- Simulação Monte Carlo (análise estatística)

## 🔬 Método Numérico

### Integrador RK4
O simulador utiliza o método de **Runge-Kutta de 4ª ordem** para resolver as equações diferenciais do movimento:

```
d²r/dt² = -GM r/|r|³
```
**Características**:
- **Precisão**: Float64 (double precision)
- **Ordem**: 4ª ordem (erro O(dt⁵))
- **Física**: Lei da Gravitação Universal de Newton
- **Conservação**: Energia e momento angular validados

## 📁 Estrutura do Código

```
simulador_orbital_asteroides_v2.py
├── PARTE 1: Imports e Constantes
│   ├── G, UA, M_SOL, M_TERRA, R_TERRA, M_LUA
│   └── RAIOS_COLISAO
├── PARTE 2: Classes Principais
│   ├── CorpoCeleste
│   │   ├── Atributos: nome, massa, posicao, velocidade
│   │   └── Métodos: salvar_estado(), energia_cinetica()
│   ├── ResultadoSimulacao
│   │   ├── corpo_colidido
│   │   └── gerar_relatorio()
│   └── SistemaGravitacional
│       ├── calcular_forca_gravitacional()
│       ├── integrador_rk4()
│       ├── simular()
│       └── detectar_colisoes_e_aproximacao()
├── PARTE 3: Funções de Configuração
│   ├── criar_sistema_base()
│   ├── criar_sistema_terra_sol()
│   ├── criar_sistema_apophis()
│   ├── criar_sistema_impacto(incluir_lua=True/False)
│   ├── criar_sistema_com_lua()
│   ├── criar_sistema_personalizado()
│   ├── criar_sistema_aleatorio()
│   └── criar_sistema_solar_completo()
├── PARTE 4: Funções de Visualização
│   ├── plotar_trajetorias()
│   ├── plotar_trajetorias_sistema_solar()
│   ├── plotar_animacao_interativa()
│   ├── plotar_distancia_temporal()
│   └── plotar_conservacao_energia()
├── PARTE 5: Simulação Monte Carlo
│   ├── simulacao_monte_carlo(massa_base, posicao_base, velocidade_base)
│   ├── plotar_resultados_monte_carlo()
│   └── plotar_trajetorias_monte_carlo()
├── PARTE 6: I/O (JSON)
│   ├── salvar_configuracao()
│   ├── carregar_configuracao()
│   └── exportar_trajetorias()
├── PARTE 7: Menu Interativo
│   ├── menu_principal()
│   └── executar_simulacao_interativa()
├── PARTE 8: Testes de Validação
│   ├── teste_conservacao_energia()
│   ├── teste_terceira_lei_kepler()
│   ├── teste_orbita_estavel()
│   ├── teste_conservacao_momento_angular()
│   └── executar_todos_testes()
├── PARTE 9: Exemplos de Uso
│   ├── exemplo_basico()
│   ├── exemplo_apophis()
│   ├── exemplo_impacto()
│   ├── exemplo_monte_carlo()
│   └── exemplo_personalizado()
└── PARTE 10: Documentação e Ajuda
    └── mostrar_ajuda()
```

## 📈 Resultados da Simulação

Exemplo de Saída (Apophis)
```
======================================================================
                    RELATÓRIO DA SIMULAÇÃO ORBITAL                    
======================================================================

INFORMAÇÕES TEMPORAIS:
  Tempo total simulado: 3.00 anos
  Número de passos: 26,298

APROXIMAÇÃO MÍNIMA:
  Distância mínima: 38371.97 km
  Distância em raios terrestres: 6.02 R⊕
  Tempo: 2.4118 anos
  Velocidade relativa: 7.18 km/s

✓ Nenhuma colisão detectada

VALIDAÇÃO FÍSICA:
  Erro relativo de energia: 1.02e-14
  ✓ Energia conservada
  Erro relativo de momento angular: 4.54e-15
  ✓ Momento angular conservado
```

Exemplo de Saída (Colisão)
```
======================================================================
                    RELATÓRIO DA SIMULAÇÃO ORBITAL                    
======================================================================

INFORMAÇÕES TEMPORAIS:
  Tempo total simulado: 0.17 anos
  Número de passos: 10,519

APROXIMAÇÃO MÍNIMA:
  Distância mínima: 2420.69 km
  Distância em raios terrestres: 0.38 R⊕
  Tempo: 0.1696 anos
  Velocidade relativa: 59.40 km/s

⚠️ COLISÃO COM A TERRA!
  Tempo: 62.0 dias
  Velocidade: 66.12 km/s
  Ângulo: 47.72°
  Energia: 1.09e+18 J
  TNT equivalente: 2.61e+02 Mt
  Raio da cratera: 22.77 km
  Corpo: Terra

VALIDAÇÃO FÍSICA:
  Erro relativo de energia: -4.35e-15
  ✓ Energia conservada
  Erro relativo de momento angular: 2.73e-15
  ✓ Momento angular conservado
```
