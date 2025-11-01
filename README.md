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
- Versão Python: instale o "simulador_orbital_asteroides_v2" e confira instruções de uso em "exemplos_v2"
- Versão Web: [Simulador de Asteroides](https://asteroidsim-kp2wuqcw.manus.space/) (em desenvolvimento)


## 📊 Funcionalidades

### Cenários Pré-configurados
- 🌍 **Terra-Sol**: Validação do integrador
- ☄️ **Apophis**: Asteroide real (aproximação 2029)
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
├── PARTE 2: Classe CorpoCeleste
│   ├── Atributos: nome, massa, posicao, velocidade
│   └── Métodos: salvar_estado(), energia_cinetica()
├── PARTE 3: Classe ResultadoSimulacao
│   ├── corpo_colidido
│   └── gerar_relatorio()
├── PARTE 4: Classe SistemaGravitacional
│   ├── calcular_forca_gravitacional()
│   ├── integrador_rk4()
│   ├── simular()
│   └── detectar_colisoes_e_aproximacao()
├── PARTE 5: Funções de Configuração
│   ├── criar_sistema_base()
│   ├── criar_sistema_terra_sol()
│   ├── criar_sistema_apophis()
│   ├── criar_sistema_impacto(incluir_lua=True/False)
│   ├── criar_sistema_com_lua()
│   └── criar_sistema_solar_completo()
├── PARTE 6: Funções de Visualização
│   ├── plotar_trajetorias()
│   ├── plotar_animacao_interativa()
│   ├── plotar_distancia_temporal()
│   └── plotar_conservacao_energia()
├── PARTE 7: Simulação Monte Carlo
│   ├── simulacao_monte_carlo()
│   ├── plotar_resultados_monte_carlo()
│   └── plotar_trajetorias_monte_carlo()
├── PARTE 8: I/O (JSON)
│   ├── salvar_configuracao()
│   ├── carregar_configuracao()
│   └── exportar_trajetorias()
├── PARTE 9: Menu Interativo
│   └── executar_simulacao_interativa()
└── PARTE 10: Testes e Documentação
    ├── executar_todos_testes()
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
  Número de passos: 26,304

APROXIMAÇÃO MÍNIMA:
  Distância mínima: 38,400.00 km
  Distância em raios terrestres: 6.03 R⊕
  Tempo da aproximação: 1.2456 anos
  Velocidade relativa: 12.45 km/s

✓ Nenhuma colisão detectada

VALIDAÇÃO FÍSICA:
  Energia inicial: -4.456789e+33 J
  Energia final: -4.456791e+33 J
  Erro relativo de energia: 4.48e-07
  ✓ Energia conservada dentro da tolerância
```

Exemplo de Saída (Colisão)
```
======================================================================
⚠️ COLISÃO COM A TERRA!
  Tempo: 45.3 dias
  Velocidade: 28.45 km/s
  Ângulo: 65.23°
  Energia: 4.05e+20 J
  TNT equivalente: 9.68e+04 Mt
  Raio da cratera: 12.34 km
  Corpo: Terra

VALIDAÇÃO FÍSICA:
  Erro relativo de energia: 3.21e-07
  ✓ Energia conservada
======================================================================
```
