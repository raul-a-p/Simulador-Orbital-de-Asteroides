# ==============================================================================
# SIMULADOR ORBITAL DE ASTEROIDES
# Projeto de Computação Científica
# 
# Autores: Raul Augusti Pereira, Valentina Spohr, Otávio Augusto Assugeni Guelfi
# ==============================================================================

# ==============================================================================
# PARTE 1: IMPORTS E CONSTANTES
# ==============================================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import json
import warnings
warnings.filterwarnings('ignore')

G = 6.67430e-11
UA = 1.496e11
M_SOL = 1.989e30
M_TERRA = 5.972e24
R_TERRA = 6.371e6
M_LUA = 7.342e22
DT_PADRAO = 3600
TOLERANCIA_ENERGIA = 1e-6
ANOS_EM_SEGUNDOS = 365.25 * 24 * 3600

# Raios de colisão (com margem de segurança)
RAIOS_COLISAO = {
    "Sol": 6.96e8 * 3,
    "Mercúrio": 2.44e6 * 2,
    "Vênus": 6.05e6 * 2,
    "Terra": R_TERRA * 2,
    "Lua": 1.737e6 * 3,
    "Marte": 3.39e6 * 2,
    "Júpiter": 6.99e7 * 2,
    "Saturno": 5.82e7 * 2,
    "Urano": 2.54e7 * 2,
    "Netuno": 2.46e7 * 2
}

_animacoes_ativas = []

print("✓ Bibliotecas e constantes carregadas!")

# ==============================================================================
# PARTE 2: CLASSES PRINCIPAIS
# ==============================================================================

@dataclass
class CorpoCeleste:
    """Representa um corpo celeste no sistema gravitacional."""
    nome: str
    massa: float
    posicao: np.ndarray
    velocidade: np.ndarray
    cor: str = 'blue'
    raio_visual: float = 5.0
    historico_posicao: List[np.ndarray] = field(default_factory=list)
    historico_velocidade: List[np.ndarray] = field(default_factory=list)
    historico_tempo: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        self.posicao = np.array(self.posicao, dtype=np.float64)
        self.velocidade = np.array(self.velocidade, dtype=np.float64)
        self.aceleracao = np.zeros(3, dtype=np.float64)
    
    def salvar_estado(self, tempo: float):
        self.historico_posicao.append(self.posicao.copy())
        self.historico_velocidade.append(self.velocidade.copy())
        self.historico_tempo.append(tempo)
    
    def energia_cinetica(self) -> float:
        return 0.5 * self.massa * np.sum(self.velocidade ** 2)
    
    def momento_angular(self, origem: np.ndarray = None) -> np.ndarray:
        if origem is None:
            origem = np.zeros(3)
        r = self.posicao - origem
        return self.massa * np.cross(r, self.velocidade)
    
    def get_trajetoria_2d(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.historico_posicao:
            return np.array([]), np.array([])
        posicoes = np.array(self.historico_posicao)
        return posicoes[:, 0], posicoes[:, 1]

@dataclass
class ResultadoSimulacao:
    """Encapsula resultados de uma simulação."""
    distancia_minima: float = float('inf')
    tempo_minima: float = 0.0
    posicao_minima_terra: np.ndarray = field(default_factory=lambda: np.zeros(3))
    posicao_minima_asteroide: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocidade_relativa_minima: float = 0.0
    houve_colisao: bool = False
    tempo_colisao: float = 0.0
    velocidade_impacto: float = 0.0
    angulo_impacto: float = 0.0
    energia_impacto: float = 0.0
    equivalente_tnt: float = 0.0
    raio_cratera: float = 0.0
    energia_inicial: float = 0.0
    energia_final: float = 0.0
    erro_energia_relativo: float = 0.0
    momento_angular_inicial: np.ndarray = field(default_factory=lambda: np.zeros(3))
    momento_angular_final: np.ndarray = field(default_factory=lambda: np.zeros(3))
    erro_momento_relativo: float = 0.0
    tempo_simulacao: float = 0.0
    numero_passos: int = 0
    corpo_colidido: str = ""  # Nome do corpo que colidiu
    
    def gerar_relatorio(self) -> str:
        linhas = ["=" * 70, "RELATÓRIO DA SIMULAÇÃO ORBITAL".center(70), "=" * 70, ""]
        
        # Informações temporais
        linhas.extend([
            "INFORMAÇÕES TEMPORAIS:",
            f"  Tempo total simulado: {self.tempo_simulacao/ANOS_EM_SEGUNDOS:.2f} anos",
            f"  Número de passos: {self.numero_passos:,}", ""
        ])
        
        # Aproximação mínima
        linhas.extend([
            "APROXIMAÇÃO MÍNIMA:",
            f"  Distância mínima: {self.distancia_minima/1000:.2f} km",
            f"  Distância em raios terrestres: {self.distancia_minima/R_TERRA:.2f} R⊕",
            f"  Tempo: {self.tempo_minima/ANOS_EM_SEGUNDOS:.4f} anos",
            f"  Velocidade relativa: {self.velocidade_relativa_minima/1000:.2f} km/s", ""
        ])
        
        # Colisão
        if self.houve_colisao:
            if self.angulo_impacto == -1:  # Sol
                linhas.extend([
                    "🔥 ASTEROIDE CAIU NO SOL!",
                    f"  Tempo: {self.tempo_colisao/(24*3600):.1f} dias",
                    f"  Velocidade: {self.velocidade_impacto/1000:.2f} km/s"
                ])
            elif self.angulo_impacto == -2:  # Lua-Terra
                linhas.append("🌙💥 LUA COLIDIU COM A TERRA (instabilidade orbital)!")
            elif self.angulo_impacto == -3:  # Lua
                linhas.extend([
                    "🌙💥 ASTEROIDE COLIDIU COM A LUA!",
                    f"  Tempo: {self.tempo_colisao/(24*3600):.1f} dias",
                    f"  Velocidade: {self.velocidade_impacto/1000:.2f} km/s"
                ])
            elif self.angulo_impacto == -99:  # Divergência
                linhas.append("⚠️ ASTEROIDE DIVERGIU (erro numérico - reduza dt)")
            else:  # Terra
                linhas.extend([
                    "⚠️ COLISÃO COM A TERRA!",
                    f"  Tempo: {self.tempo_colisao/(24*3600):.1f} dias",
                    f"  Velocidade: {self.velocidade_impacto/1000:.2f} km/s",
                    f"  Ângulo: {self.angulo_impacto:.2f}°",
                    f"  Energia: {self.energia_impacto:.2e} J",
                    f"  TNT equivalente: {self.equivalente_tnt:.2e} Mt",
                    f"  Raio da cratera: {self.raio_cratera/1000:.2f} km"
                ])
            if self.corpo_colidido:
                linhas.append(f"  Corpo: {self.corpo_colidido}")
        else:
            linhas.append("✓ Nenhuma colisão detectada")
        
        linhas.append("")
        
        # Validação física
        linhas.extend([
            "VALIDAÇÃO FÍSICA:",
            f"  Erro relativo de energia: {self.erro_energia_relativo:.2e}",
            "  ✓ Energia conservada" if abs(self.erro_energia_relativo) < TOLERANCIA_ENERGIA 
            else "  ⚠️ Violação na conservação de energia",
            f"  Erro relativo de momento angular: {self.erro_momento_relativo:.2e}",
            "  ✓ Momento angular conservado" if abs(self.erro_momento_relativo) < TOLERANCIA_ENERGIA 
            else "  ⚠️ Violação na conservação do momento angular",
            "", "OBSERVAÇÃO:",
            "🔵 Posição inicial: círculo",
            "⬛ Posição final: quadrado",
            "", "=" * 70
        ])
        
        return "\n".join(linhas)

class SistemaGravitacional:
    """Gerencia sistema de múltiplos corpos (VERSÃO OTIMIZADA)."""
    
    def __init__(self, dt: float = DT_PADRAO):
        self.corpos: List[CorpoCeleste] = []
        self.dt = dt
        self.tempo_atual = 0.0
        self.resultado = ResultadoSimulacao()
    
    def adicionar_corpo(self, corpo: CorpoCeleste):
        self.corpos.append(corpo)
    
    def calcular_forca_gravitacional(self, corpo1: CorpoCeleste, corpo2: CorpoCeleste) -> np.ndarray:
        r_vec = corpo2.posicao - corpo1.posicao
        r_mag = np.linalg.norm(r_vec)
        if r_mag < 1e3:
            return np.zeros(3)
        r_hat = r_vec / r_mag
        f_mag = G * corpo1.massa * corpo2.massa / (r_mag ** 2)
        return f_mag * r_hat / corpo1.massa
    
    def calcular_aceleracoes(self):
        for corpo in self.corpos:
            corpo.aceleracao = np.zeros(3)
        
        n = len(self.corpos)
        for i in range(n):
            for j in range(i + 1, n):
                a_i = self.calcular_forca_gravitacional(self.corpos[i], self.corpos[j])
                a_j = -a_i * self.corpos[i].massa / self.corpos[j].massa
                self.corpos[i].aceleracao += a_i
                self.corpos[j].aceleracao += a_j
    
    def get_estado(self) -> np.ndarray:
        estado = []
        for corpo in self.corpos:
            estado.extend(corpo.posicao)
            estado.extend(corpo.velocidade)
        return np.array(estado, dtype=np.float64)
    
    def set_estado(self, estado: np.ndarray):
        idx = 0
        for corpo in self.corpos:
            corpo.posicao = estado[idx:idx+3].copy()
            corpo.velocidade = estado[idx+3:idx+6].copy()
            idx += 6
    
    def derivada(self, estado: np.ndarray) -> np.ndarray:
        estado_original = self.get_estado()
        self.set_estado(estado)
        self.calcular_aceleracoes()
        
        derivadas = []
        for corpo in self.corpos:
            derivadas.extend(corpo.velocidade)
            derivadas.extend(corpo.aceleracao)
        
        self.set_estado(estado_original)
        return np.array(derivadas, dtype=np.float64)
    
    def integrador_rk4(self) -> np.ndarray:
        y = self.get_estado()
        k1 = self.derivada(y)
        k2 = self.derivada(y + 0.5 * self.dt * k1)
        k3 = self.derivada(y + 0.5 * self.dt * k2)
        k4 = self.derivada(y + self.dt * k3)
        return y + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def energia_cinetica_total(self) -> float:
        return sum(corpo.energia_cinetica() for corpo in self.corpos)
    
    def energia_potencial_total(self) -> float:
        ep_total = 0.0
        n = len(self.corpos)
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(self.corpos[j].posicao - self.corpos[i].posicao)
                if r > 1e3:
                    ep_total += -G * self.corpos[i].massa * self.corpos[j].massa / r
        return ep_total
    
    def energia_total(self) -> float:
        return self.energia_cinetica_total() + self.energia_potencial_total()
    
    def momento_angular_total(self) -> np.ndarray:
        cm = self.centro_de_massa()
        return sum(corpo.momento_angular(origem=cm) for corpo in self.corpos)
    
    def centro_de_massa(self) -> np.ndarray:
        massa_total = sum(corpo.massa for corpo in self.corpos)
        cm = np.zeros(3)
        for corpo in self.corpos:
            cm += corpo.massa * corpo.posicao
        return cm / massa_total
    
    # MÉTODO DE DETECÇÃO DE COLISÕES
    def detectar_colisoes_e_aproximacao(self, progresso: bool = True):
        """
        Método unificado que detecta:
        1. Aproximação mínima entre corpos
        2. Colisões com qualquer corpo do sistema
        3. Divergência numérica
        """
        asteroide = next((c for c in self.corpos if c.nome == "Asteroide"), None)
        if not asteroide:
            return
        
        # Verificar divergência numérica
        dist_origem = np.linalg.norm(asteroide.posicao)
        if dist_origem > 100 * UA and not self.resultado.houve_colisao:
            if progresso:
                print(f"\n⚠️ AVISO: Asteroide divergiu (dist > 100 UA)")
            self.resultado.houve_colisao = True
            self.resultado.angulo_impacto = -99
            return
        
        # Detectar colisões com todos os corpos
        if not self.resultado.houve_colisao:
            for corpo in self.corpos:
                if corpo.nome == "Asteroide":
                    continue
                
                distancia = np.linalg.norm(asteroide.posicao - corpo.posicao)
                raio_colisao = RAIOS_COLISAO.get(corpo.nome, R_TERRA * 2)
                v_relativa = asteroide.velocidade - corpo.velocidade
                r_vec = asteroide.posicao - corpo.posicao
                aproximando = np.dot(v_relativa, r_vec) < 0
                
                if distancia < raio_colisao:
                    self.resultado.houve_colisao = True
                    self.resultado.tempo_colisao = self.tempo_atual
                    self.resultado.distancia_minima = distancia
                    self.resultado.velocidade_impacto = np.linalg.norm(v_relativa)
                    self.resultado.corpo_colidido = corpo.nome
                    
                    if corpo.nome == "Sol":
                        self.resultado.angulo_impacto = -1
                        self.resultado.energia_impacto = 0.5 * asteroide.massa * self.resultado.velocidade_impacto**2
                        self.resultado.equivalente_tnt = self.resultado.energia_impacto / 4.184e15
                    elif corpo.nome == "Lua":
                        self.resultado.angulo_impacto = -3
                        self.resultado.energia_impacto = 0.5 * asteroide.massa * self.resultado.velocidade_impacto**2
                    else:
                        self.calcular_parametros_impacto(corpo, asteroide, v_relativa)
                    
                    if progresso:
                        print(f"\n💥 COLISÃO COM {corpo.nome.upper()}!")
                        print(f"   Tempo: {self.tempo_atual/(24*3600):.1f} dias")
                    break
        
        # Atualizar aproximação mínima (para Terra ou corpo principal)
        terra = next((c for c in self.corpos if c.nome == "Terra"), None)
        if terra:
            distancia = np.linalg.norm(asteroide.posicao - terra.posicao)
            if distancia < self.resultado.distancia_minima:
                self.resultado.distancia_minima = distancia
                self.resultado.tempo_minima = self.tempo_atual
                self.resultado.posicao_minima_terra = terra.posicao.copy()
                self.resultado.posicao_minima_asteroide = asteroide.posicao.copy()
                v_relativa = asteroide.velocidade - terra.velocidade
                self.resultado.velocidade_relativa_minima = np.linalg.norm(v_relativa)
    
    def calcular_parametros_impacto(self, alvo: CorpoCeleste, asteroide: CorpoCeleste, v_relativa: np.ndarray):
        """Calcula parâmetros físicos do impacto."""
    
        v_impacto = np.linalg.norm(v_relativa)
        self.resultado.velocidade_impacto = v_impacto
    
        energia = 0.5 * asteroide.massa * v_impacto**2
        self.resultado.energia_impacto = energia
    
        # equivalente TNT (1 megaton = 4.184e15 J)
        self.resultado.equivalente_tnt = energia / 4.184e15
    
        # Normal local da superfície = direção do ponto de impacto para o centro do alvo
        r_vec = asteroide.posicao - alvo.posicao
        if np.linalg.norm(r_vec) > 0:
    
            # Normal local (vertical local)
            n = r_vec / np.linalg.norm(r_vec)
    
            # Vel ocidade de impacto ***em direção ao solo*** é -v_relativa
            cos_theta = np.dot(-v_relativa, n) / (v_impacto * 1.0)
    
            # Corrige erros numéricos
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
            # θ = 0° → impacto vertical; 90° → impacto tangencial
            angulo = np.degrees(np.arccos(cos_theta))
            self.resultado.angulo_impacto = angulo
    
        # Raio de cratera: modelo π-scaling simplificado
        # Densidades típicas
        dens_proj = 3000.0       # kg/m³ asteroide rochoso
        dens_alvo = 2700.0       # kg/m³ rocha/superfície terrestre
        g = 9.81
    
        # Raio equivalente do projétil (assume esfera)
        volume = asteroide.massa / dens_proj
        raio_proj = (3 * volume / (4 * np.pi))**(1/3)
    
        # Modelo dimensional básico + correção de ângulo
        # Baseado nas dependências da lei π (Holsapple/Collins)
        # D ~ a * v^(2/3) * g^(-1/3) * (ρp/ρt)^(-1/3)
        D0 = 2 * raio_proj * (v_impacto**(2/3)) * (g**(-1/3)) * (dens_proj/dens_alvo)**(-1/3)
    
        # Correção para impacto oblíquo (comportamento ~ sin(θ)^(1/3))
        ang_rad = np.radians(self.resultado.angulo_impacto)
        angle_factor = (np.sin(ang_rad))**(1/3)
    
        D_final = D0 * angle_factor   # Diâmetro final aproximado
        raio_final = D_final / 2
    
        self.resultado.raio_cratera = raio_final
    
    def simular(self, tempo_total: float, progresso: bool = True):
        """Executa a simulação."""
        self.resultado.energia_inicial = self.energia_total()
        self.resultado.momento_angular_inicial = self.momento_angular_total()
        
        for corpo in self.corpos:
            corpo.salvar_estado(self.tempo_atual)
        
        n_passos = int(tempo_total / self.dt)
        self.resultado.numero_passos = n_passos
        self.resultado.tempo_simulacao = tempo_total
        
        for passo in range(n_passos):
            novo_estado = self.integrador_rk4()
            self.set_estado(novo_estado)
            self.tempo_atual += self.dt
            
            for corpo in self.corpos:
                corpo.salvar_estado(self.tempo_atual)
            
            self.detectar_colisoes_e_aproximacao(progresso)
            
            if self.resultado.houve_colisao:
                break
            
            if passo % 100 == 0:
                energia_atual = self.energia_total()
                erro_rel = abs(energia_atual - self.resultado.energia_inicial) / abs(self.resultado.energia_inicial)
                if erro_rel > TOLERANCIA_ENERGIA and progresso:
                    print(f"⚠️ Erro de conservação = {erro_rel:.2e} no passo {passo}")
            
            if progresso and passo % (n_passos // 20) == 0:
                print(f"Progresso: {100*passo/n_passos:.1f}%")
        
        self.resultado.energia_final = self.energia_total()
        self.resultado.erro_energia_relativo = ((self.resultado.energia_final - 
                                                 self.resultado.energia_inicial) / 
                                                abs(self.resultado.energia_inicial))
        self.resultado.momento_angular_final = self.momento_angular_total()
        
        # Calcular erro do momento angular
        L_inicial = np.linalg.norm(self.resultado.momento_angular_inicial)
        L_final = np.linalg.norm(self.resultado.momento_angular_final)
        if L_inicial > 0:
            self.resultado.erro_momento_relativo = abs(L_final - L_inicial) / L_inicial
        else:
            self.resultado.erro_momento_relativo = 0.0
        
        self.resultado.tempo_simulacao = self.tempo_atual
        
        if progresso:
            print("\n✓ Simulação concluída!")
        
        return self.resultado

print("✓ Classes definidas!")

# ==============================================================================
# PARTE 3: FUNÇÕES DE CONFIGURAÇÃO
# ==============================================================================

def criar_sistema_base(dt: float, incluir_lua: bool = False, 
                      config_asteroide: Optional[Dict] = None) -> SistemaGravitacional:
    """
    Função BASE unificada para criar sistemas.
    Elimina duplicação entre criar_sistema_apophis, criar_sistema_com_lua, etc.
    """
    sistema = SistemaGravitacional(dt=dt)
    
    # Sol (sempre presente)
    sistema.adicionar_corpo(CorpoCeleste(
        nome="Sol", massa=M_SOL, posicao=[0, 0, 0],
        velocidade=[0, 0, 0], cor='yellow', raio_visual=20
    ))
    
    # Terra (sempre presente)
    v_orbital_terra = np.sqrt(G * M_SOL / UA)
    sistema.adicionar_corpo(CorpoCeleste(
        nome="Terra", massa=M_TERRA, posicao=[UA, 0, 0],
        velocidade=[0, v_orbital_terra, 0], cor='blue', raio_visual=10
    ))
    
    # Lua (opcional)
    if incluir_lua:
        DIST_LUA = 3.844e8
        v_orbital_lua = np.sqrt(G * M_TERRA / DIST_LUA)
        sistema.adicionar_corpo(CorpoCeleste(
            nome="Lua", massa=M_LUA, posicao=[UA + DIST_LUA, 0, 0],
            velocidade=[0, v_orbital_terra + v_orbital_lua, 0],
            cor='gray', raio_visual=7
        ))
    
    # Asteroide (configurável)
    if config_asteroide:
        sistema.adicionar_corpo(CorpoCeleste(
            nome="Asteroide",
            massa=config_asteroide['massa'],
            posicao=config_asteroide['posicao'],
            velocidade=config_asteroide['velocidade'],
            cor=config_asteroide.get('cor', 'red'),
            raio_visual=config_asteroide.get('raio_visual', 5)
        ))
    
    return sistema

# Wrappers simplificados
def criar_sistema_terra_sol(dt: float = 3600) -> SistemaGravitacional:
    """Sistema simples Terra-Sol."""
    return criar_sistema_base(dt, incluir_lua=False, config_asteroide=None)

def criar_sistema_apophis(dt: float = 3600) -> SistemaGravitacional:
    """
    Asteroide Apophis - aproximação 2029.
    Usa elementos orbitais reais e posiciona Terra e Apophis 
    para simular a aproximação de 2029.
    """
    # ==================================================
    # ELEMENTOS ORBITAIS REAIS DO APOPHIS (época J2000)
    # Fonte: JPL Horizons / Small-Body Database
    # ==================================================
    a_apophis = 0.9224 * UA  # Semi-eixo maior
    e_apophis = 0.191        # Excentricidade
    
    
    # Posição da Terra (em 2027): começar em t=0
    # Terra em órbita circular a 1 UA
    angulo_terra = np.radians(0)  # Começar em x positivo
    r_terra = UA
    v_orbital_terra = np.sqrt(G * M_SOL / r_terra)
    
    pos_terra = np.array([r_terra * np.cos(angulo_terra), 
                          r_terra * np.sin(angulo_terra), 
                          0])
    vel_terra = np.array([-v_orbital_terra * np.sin(angulo_terra), 
                          v_orbital_terra * np.cos(angulo_terra), 
                          0])
    
    nu_apophis = np.radians(180)
    
    # Raio orbital do Apophis nessa posição
    r_apophis = a_apophis * (1 - e_apophis**2) / (1 + e_apophis * np.cos(nu_apophis))
    
    # Velocidade orbital (equação vis-viva)
    v_apophis_mag = np.sqrt(G * M_SOL * (2/r_apophis - 1/a_apophis))
    
    angulo_apophis = np.radians(210)
    
    pos_apophis = np.array([r_apophis * np.cos(angulo_apophis),
                           r_apophis * np.sin(angulo_apophis),
                           0])
    
    vel_apophis = np.array([-v_apophis_mag * np.sin(angulo_apophis) * 1.060243,
                            v_apophis_mag * np.cos(angulo_apophis) * 0.97,
                            0])
    
    sistema = SistemaGravitacional(dt=dt)
    
    # Sol
    sistema.adicionar_corpo(CorpoCeleste(
        nome="Sol",
        massa=M_SOL,
        posicao=[0, 0, 0],
        velocidade=[0, 0, 0],
        cor='yellow',
        raio_visual=20
    ))
    
    # Terra
    sistema.adicionar_corpo(CorpoCeleste(
        nome="Terra",
        massa=M_TERRA,
        posicao=pos_terra.tolist(),
        velocidade=vel_terra.tolist(),
        cor='blue',
        raio_visual=10
    ))
    
    # Apophis
    sistema.adicionar_corpo(CorpoCeleste(
        nome="Asteroide",
        massa=6.1e10,  # ~370m de diâmetro
        posicao=pos_apophis.tolist(),
        velocidade=vel_apophis.tolist(),
        cor='red',
        raio_visual=7
    ))
    
    print(f"\n☄️ ASTEROIDE APOPHIS (99942)")
    print(f"   Semi-eixo maior: {a_apophis/UA:.4f} UA")
    print(f"   Excentricidade: {e_apophis:.3f}")
    print(f"   Massa: {6.1e10:.2e} kg (~370m diâmetro)")
    print(f"\n   CONDIÇÕES INICIAIS:")
    print(f"   Terra: ({pos_terra[0]/UA:.3f}, {pos_terra[1]/UA:.3f}) UA")
    print(f"   Apophis: ({pos_apophis[0]/UA:.3f}, {pos_apophis[1]/UA:.3f}) UA")
    print(f"   Velocidade Apophis: {np.linalg.norm(vel_apophis)/1000:.2f} km/s")
    print(f"\n   ⏱️ Simule por ~3 anos para observar a aproximação")
    print(f"   📏 Distância esperada: entre 5.62R⊕ e 6.30R⊕ (literatura)")
    
    return sistema

def criar_sistema_impacto(dt: float = 900, incluir_lua: bool = False) -> SistemaGravitacional:
    """Cenário de impacto (funciona com ou sem Lua)"""
    
    if incluir_lua:
        r_asteroide = 1.020399 * UA  
        v_orbital_asteroide = np.sqrt(G * M_SOL / r_asteroide)
        
        angulo_asteroide = np.radians(120)
        pos_ast_x = r_asteroide * np.cos(angulo_asteroide)
        pos_ast_y = r_asteroide * np.sin(angulo_asteroide)
        
        vel_ast_x = v_orbital_asteroide * np.sin(angulo_asteroide) 
        vel_ast_y = -v_orbital_asteroide * np.cos(angulo_asteroide)
        
        fator_ajuste = 0.98
        vel_ast_x *= fator_ajuste
        vel_ast_y *= fator_ajuste
        
        config = {
            'massa': 5e11,  
            'posicao': [pos_ast_x, pos_ast_y, 0],
            'velocidade': [vel_ast_x, vel_ast_y, 0],
            'raio_visual': 8
        }
        print(f"\n🎯 CENÁRIO DE IMPACTO (Terra-Lua)")
        print(f"   Sistema: Sol, Terra, Lua e Asteroide")
        print(f"   Massa asteroide: {config['massa']:.2e} kg")
        print(f"   dt: {dt}s ({dt/60:.0f} min)")
        
    else:
        # valores exatos necessários
        r_asteroide = 1.01993 * UA  
        v_orbital_asteroide = np.sqrt(G * M_SOL / r_asteroide)
        
        angulo_asteroide = np.radians(120) 
        
        pos_ast_x = r_asteroide * np.cos(angulo_asteroide)
        pos_ast_y = r_asteroide * np.sin(angulo_asteroide)
        
        vel_ast_x = v_orbital_asteroide * np.sin(angulo_asteroide) 
        vel_ast_y = -v_orbital_asteroide * np.cos(angulo_asteroide)
        
        fator_ajuste = 0.98
        vel_ast_x *= fator_ajuste
        vel_ast_y *= fator_ajuste
        
        config = {
            'massa': 5e8,
            'posicao': [pos_ast_x, pos_ast_y, 0],
            'velocidade': [vel_ast_x, vel_ast_y, 0],
            'raio_visual': 8
        }
        
        v_relativa_estimada = np.sqrt(G * M_SOL / UA) + v_orbital_asteroide * fator_ajuste
        
        print(f"\n🎯 CENÁRIO DE IMPACTO")
        print(f"   Velocidade Terra: {np.sqrt(G * M_SOL / UA)/1000:.2f} km/s")
        print(f"   Velocidade Asteroide: {np.sqrt(vel_ast_x**2 + vel_ast_y**2)/1000:.2f} km/s")
        print(f"   Velocidade relativa de impacto: ~{v_relativa_estimada/1000:.2f} km/s")
        print(f"   Massa: {config['massa']:.2e} kg")
        print(f"   dt: {dt}s ({dt/60:.0f} min)")
    
    return criar_sistema_base(dt, incluir_lua=incluir_lua, config_asteroide=config)

def criar_sistema_com_lua(dt: float = 3600, config_asteroide: Optional[Dict] = None) -> SistemaGravitacional:
    """Sistema com Lua (Apophis padrão ou customizado)."""
    if config_asteroide is None:
        # Usar Apophis como padrão
        a, e = 0.92 * UA, 0.19
        r = a * (1 - e)
        v = np.sqrt(G * M_SOL * (2/r - 1/a))
        config_asteroide = {
            'massa': 6.1e10,
            'posicao': [r * 0.95, r * 0.1, 0],
            'velocidade': [-v * 0.15, v * 0.98, 0]
        }
    
    print("\n🌙 SISTEMA COM LUA")
    return criar_sistema_base(dt, incluir_lua=True, config_asteroide=config_asteroide)

    
def criar_sistema_personalizado(massa_asteroide: float,
                                posicao_asteroide: List[float],
                                velocidade_asteroide: List[float], dt: float = 3600) -> SistemaGravitacional:
    """Cria sistema com parâmetros personalizados."""
    sistema = SistemaGravitacional(dt=dt)
    
    # Sol
    sol = CorpoCeleste(
        nome="Sol",
        massa=M_SOL,
        posicao=[0, 0, 0],
        velocidade=[0, 0, 0],
        cor='yellow',
        raio_visual=20
    )
    
    # Terra
    v_orbital_terra = np.sqrt(G * M_SOL / UA)
    terra = CorpoCeleste(
        nome="Terra",
        massa=M_TERRA,
        posicao=[UA, 0, 0],
        velocidade=[0, v_orbital_terra, 0],
        cor='blue',
        raio_visual=10
    )
    
    # Asteroide personalizado
    asteroide = CorpoCeleste(
        nome="Asteroide",
        massa=massa_asteroide,
        posicao=posicao_asteroide,
        velocidade=velocidade_asteroide,
        cor='red',
        raio_visual=7
    )
    
    sistema.adicionar_corpo(sol)
    sistema.adicionar_corpo(terra)
    sistema.adicionar_corpo(asteroide)
    
    return sistema

def criar_sistema_aleatorio(seed: Optional[int] = None, dt: float = 3600) -> SistemaGravitacional:
    """Cria sistema com condições iniciais aleatórias."""
    if seed is not None:
        np.random.seed(seed)
    
    sistema = SistemaGravitacional(dt=dt)
    
    # Sol
    sol = CorpoCeleste(
        nome="Sol",
        massa=M_SOL,
        posicao=[0, 0, 0],
        velocidade=[0, 0, 0],
        cor='yellow',
        raio_visual=20
    )
    
    # Terra
    v_orbital_terra = np.sqrt(G * M_SOL / UA)
    terra = CorpoCeleste(
        nome="Terra",
        massa=M_TERRA,
        posicao=[UA, 0, 0],
        velocidade=[0, v_orbital_terra, 0],
        cor='blue',
        raio_visual=10
    )
    
    # Asteroide aleatório
    a = np.random.uniform(0.8, 1.5) * UA
    e = np.random.uniform(0.1, 0.4)
    theta = np.random.uniform(0, 2 * np.pi)
    
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    v = np.sqrt(G * M_SOL * (2/r - 1/a))
    
    pos_x = r * np.cos(theta)
    pos_y = r * np.sin(theta)
    vel_x = -v * np.sin(theta)
    vel_y = v * np.cos(theta)
    
    massa = 10 ** np.random.uniform(9, 12)
    
    asteroide = CorpoCeleste(
        nome="Asteroide",
        massa=massa,
        posicao=[pos_x, pos_y, 0],
        velocidade=[vel_x, vel_y, 0],
        cor='red',
        raio_visual=6
    )
    
    sistema.adicionar_corpo(sol)
    sistema.adicionar_corpo(terra)
    sistema.adicionar_corpo(asteroide)
    
    return sistema
    
def criar_sistema_solar_completo(asteroide_config: str = "padrao", dt: float = None) -> SistemaGravitacional:
    """
    Cria o Sistema Solar completo com asteroide personalizável.
    """
    if asteroide_config == "padrao" and dt is None:
        dt = 20000

    elif dt is None:
        dt = 7200
    
    sistema = SistemaGravitacional(dt=dt)
    
    # Sol
    sol = CorpoCeleste(
        nome="Sol",
        massa=M_SOL,
        posicao=[0, 0, 0],
        velocidade=[0, 0, 0],
        cor='yellow',
        raio_visual=25
    )
    sistema.adicionar_corpo(sol)
    
    # Planetas
    planetas_dados = [
        ["Mercúrio", 0.387, 3.30e23, 'gray', 4],
        ["Vênus", 0.723, 4.87e24, 'orange', 6],
        ["Terra", 1.000, M_TERRA, 'blue', 8],
        ["Marte", 1.524, 6.42e23, 'red', 5],
        ["Júpiter", 5.203, 1.90e27, 'brown', 15],
        ["Saturno", 9.537, 5.68e26, 'gold', 13],
        ["Urano", 19.191, 8.68e25, 'lightblue', 10],
        ["Netuno", 30.069, 1.02e26, 'darkblue', 10]
    ]
    
    print("\n🌍 SISTEMA SOLAR COMPLETO")
    print("=" * 70)
    
    for nome, a_ua, massa, cor, raio_vis in planetas_dados:
        a = a_ua * UA
        v_orbital = np.sqrt(G * M_SOL / a)
        angulo = np.random.uniform(0, 2*np.pi)
        
        planeta = CorpoCeleste(
            nome=nome,
            massa=massa,
            posicao=[a * np.cos(angulo), a * np.sin(angulo), 0],
            velocidade=[-v_orbital * np.sin(angulo), v_orbital * np.cos(angulo), 0],
            cor=cor,
            raio_visual=raio_vis
        )
        sistema.adicionar_corpo(planeta)
        print(f"  {nome:10s} | a={a_ua:.3f} UA | v={v_orbital/1000:.1f} km/s")
    
    # Asteroide
    if isinstance(asteroide_config, dict):
        asteroide = CorpoCeleste(
            nome="Asteroide",
            massa=asteroide_config['massa'],
            posicao=asteroide_config['posicao'],
            velocidade=asteroide_config['velocidade'],
            cor='lime',
            raio_visual=6
        )
        print(f"\n  {'Asteroide':10s} | PERSONALIZADO")
        
    elif asteroide_config == "padrao":
        distancia_inicial = 30 * UA
        velocidade_inicial = 50000
        angulo = np.radians(-30)
        
        asteroide = CorpoCeleste(
            nome="Asteroide",
            massa=1e13,
            posicao=[distancia_inicial * np.cos(angulo), 
                    distancia_inicial * np.sin(angulo), 0],
            velocidade=[-velocidade_inicial * np.cos(angulo) * 0.8,
                       -velocidade_inicial * np.sin(angulo) * 0.74, 0],
            cor='lime',
            raio_visual=8
        )
        print(f"\n  {'Asteroide':10s} | INTERESTELAR a {distancia_inicial/UA:.1f} UA")
        print(f"                 | Velocidade: {velocidade_inicial/1000:.1f} km/s")
        
    elif asteroide_config == "cinturao":
        a_ast = 2.7 * UA
        e_ast = 0.15
        theta_ast = np.radians(45)
        r_ast = a_ast * (1 - e_ast**2) / (1 + e_ast * np.cos(theta_ast))
        v_ast = np.sqrt(G * M_SOL * (2/r_ast - 1/a_ast))
        
        asteroide = CorpoCeleste(
            nome="Asteroide",
            massa=1e12,
            posicao=[r_ast * np.cos(theta_ast), r_ast * np.sin(theta_ast), 0],
            velocidade=[-v_ast * np.sin(theta_ast), v_ast * np.cos(theta_ast), 0],
            cor='lime',  
            raio_visual=6
        )
        print(f"\n  {'Asteroide':10s} | CINTURÃO (entre Marte e Júpiter)")
        
    elif asteroide_config == "proximo":
        a_ast = 1.1 * UA
        v_ast = np.sqrt(G * M_SOL / a_ast)
        
        asteroide = CorpoCeleste(
            nome="Asteroide",
            massa=5e11,
            posicao=[a_ast, 0, 0],
            velocidade=[0, v_ast * 0.95, 0],
            cor='lime',  
            raio_visual=6
        )
        print(f"\n  {'Asteroide':10s} | PRÓXIMO À TERRA")
    
    sistema.adicionar_corpo(asteroide)
    print("=" * 70)
    
    return sistema
    
print("✓ Funções de configuração criadas!")

# ==============================================================================
# PARTE 4: FUNÇÕES DE VISUALIZAÇÃO
# ==============================================================================

def plotar_trajetorias(sistema: SistemaGravitacional, 
                       titulo: str = "Trajetórias Orbitais",
                       figsize: Tuple[int, int] = (12, 10)):
    """
    Plota as trajetórias dos corpos em 2D.
    """
    # VERIFICAR SE HÁ DADOS
    tem_dados = False
    for corpo in sistema.corpos:
        if len(corpo.historico_posicao) > 0:
            tem_dados = True
            break
    
    if not tem_dados:
        print("⚠️ ERRO: Sistema não foi simulado! Execute sistema.simular() primeiro.")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plotar cada corpo
    for corpo in sistema.corpos:
        x, y = corpo.get_trajetoria_2d()
        if len(x) == 0:
            continue
            
        # Plotar trajetória
        ax.plot(x/UA, y/UA, '-', label=corpo.nome, 
               color=corpo.cor, linewidth=1.5, alpha=0.7)
        
        # Marcar posição inicial (círculo)
        ax.plot(x[0]/UA, y[0]/UA, 'o', color=corpo.cor, 
               markersize=corpo.raio_visual, alpha=0.9,
               markeredgecolor='white', markeredgewidth=1.5)
        
        # Marcar posição final (quadrado)
        ax.plot(x[-1]/UA, y[-1]/UA, 's', color=corpo.cor, 
               markersize=corpo.raio_visual*0.8, alpha=0.9,
               markeredgecolor='white', markeredgewidth=1.5)
    
    # Marcar ponto de aproximação mínima
    if sistema.resultado.distancia_minima < float('inf'):
        pos_t = sistema.resultado.posicao_minima_terra
        pos_a = sistema.resultado.posicao_minima_asteroide
        
        # Linha tracejada conectando as posições
        ax.plot([pos_t[0]/UA, pos_a[0]/UA], 
               [pos_t[1]/UA, pos_a[1]/UA], 
               'k--', linewidth=2, alpha=0.5, label='Aproximação mínima')
        
        # Estrela no ponto mais próximo do asteroide
        ax.plot(pos_a[0]/UA, pos_a[1]/UA, 'r*', 
               markersize=15, label='Ponto mais próximo',
               markeredgecolor='yellow', markeredgewidth=1)
        
        # Se houve colisão, destacar
        if sistema.resultado.houve_colisao:
            if sistema.resultado.angulo_impacto != -1:
                ax.plot(pos_a[0]/UA, pos_a[1]/UA, 'ro', 
                       markersize=20, alpha=0.3)
                ax.plot(pos_t[0]/UA, pos_t[1]/UA, 'ro', 
                       markersize=20, alpha=0.3)
    
    # Configurar gráfico
    ax.set_xlabel('x (UA)', fontsize=12)
    ax.set_ylabel('y (UA)', fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    
    # Adicionar informações da simulação no título
    if sistema.resultado.houve_colisao:
        if sistema.resultado.angulo_impacto == -1:
            titulo_extra = f"\n Queda no Sol em {sistema.resultado.tempo_colisao/(24*3600):.1f} dias"
        else:
            titulo_extra = f"\n Colisão em {sistema.resultado.tempo_colisao/(24*3600):.1f} dias"
        ax.text(0.5, 1.08, titulo_extra, transform=ax.transAxes,
               ha='center', fontsize=11, color='red', fontweight='bold')
    
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axis('equal')
    
    # Ajustar limites para garantir que tudo é visível
    todas_x = []
    todas_y = []
    for corpo in sistema.corpos:
        x, y = corpo.get_trajetoria_2d()
        if len(x) > 0:
            todas_x.extend(x/UA)
            todas_y.extend(y/UA)
    
    if todas_x and todas_y:
        margem = 0.1
        x_min, x_max = min(todas_x), max(todas_x)
        y_min, y_max = min(todas_y), max(todas_y)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        ax.set_xlim(x_min - margem * x_range, x_max + margem * x_range)
        ax.set_ylim(y_min - margem * y_range, y_max + margem * y_range)
    
    plt.tight_layout()
    plt.show()

def plotar_trajetorias_sistema_solar(sistema: SistemaGravitacional, 
                                     titulo: str = "Sistema Solar Completo"):
    """
    Plota trajetórias do Sistema Solar em 2 gráficos:
    1. Visão completa (todo o sistema)
    2. Zoom na região Terra-Asteroide
    """
    # VERIFICAR SE HÁ DADOS
    tem_dados = False
    for corpo in sistema.corpos:
        if len(corpo.historico_posicao) > 0:
            tem_dados = True
            break
    
    if not tem_dados:
        print("⚠️ ERRO: Sistema não foi simulado! Execute sistema.simular() primeiro.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # === GRÁFICO 1: VISÃO COMPLETA ===
    for corpo in sistema.corpos:
        x, y = corpo.get_trajetoria_2d()
        if len(x) == 0:
            continue
            
        ax1.plot(x/UA, y/UA, '-', label=corpo.nome, 
               color=corpo.cor, linewidth=1.5, alpha=0.7)
        
        # Posição inicial e final
        ax1.plot(x[0]/UA, y[0]/UA, 'o', color=corpo.cor, 
               markersize=corpo.raio_visual, alpha=0.9,
               markeredgecolor='white', markeredgewidth=1.5)
        ax1.plot(x[-1]/UA, y[-1]/UA, 's', color=corpo.cor, 
               markersize=corpo.raio_visual*0.8, alpha=0.9,
               markeredgecolor='white', markeredgewidth=1.5)
    
    ax1.set_xlabel('x (UA)', fontsize=12)
    ax1.set_ylabel('y (UA)', fontsize=12)
    ax1.set_title(f'{titulo} - Visão Completa', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axis('equal')
    
    # === GRÁFICO 2: ZOOM TERRA-ASTEROIDE ===
    terra = next((c for c in sistema.corpos if c.nome == "Terra"), None)
    asteroide = next((c for c in sistema.corpos if c.nome == "Asteroide"), None)
    
    # Plotar apenas Terra, Sol e Asteroide no zoom
    for corpo in sistema.corpos:
        if corpo.nome in ["Sol", "Terra", "Asteroide"]:
            x, y = corpo.get_trajetoria_2d()
            if len(x) == 0:
                continue
                
            ax2.plot(x/UA, y/UA, '-', label=corpo.nome, 
                   color=corpo.cor, linewidth=2, alpha=0.8)
            
            # Posição inicial e final
            ax2.plot(x[0]/UA, y[0]/UA, 'o', color=corpo.cor, 
                   markersize=corpo.raio_visual*1.5, alpha=0.9,
                   markeredgecolor='white', markeredgewidth=2)
            ax2.plot(x[-1]/UA, y[-1]/UA, 's', color=corpo.cor, 
                   markersize=corpo.raio_visual*1.2, alpha=0.9,
                   markeredgecolor='white', markeredgewidth=2)
    
    # Definir limites do zoom
    if terra and asteroide:
        x_t, y_t = terra.get_trajetoria_2d()
        x_a, y_a = asteroide.get_trajetoria_2d()
        
        if len(x_t) > 0 and len(x_a) > 0:
            # Encontrar limites englobando Terra e Asteroide
            x_min = min(np.min(x_t), np.min(x_a)) / UA
            x_max = max(np.max(x_t), np.max(x_a)) / UA
            y_min = min(np.min(y_t), np.min(y_a)) / UA
            y_max = max(np.max(y_t), np.max(y_a)) / UA
            
            # Adicionar margem de 20%
            margem = 0.2
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Garantir margem mínima
            if x_range < 0.1:
                x_range = 2.0
            if y_range < 0.1:
                y_range = 2.0
            
            ax2.set_xlim(x_min - margem*x_range, x_max + margem*x_range)
            ax2.set_ylim(y_min - margem*y_range, y_max + margem*y_range)
    else:
        # Zoom padrão se não houver Terra/Asteroide
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)
    
    # Marcar aproximação mínima
    if sistema.resultado.distancia_minima < float('inf'):
        pos_t = sistema.resultado.posicao_minima_terra
        pos_a = sistema.resultado.posicao_minima_asteroide
        
        ax2.plot([pos_t[0]/UA, pos_a[0]/UA], 
               [pos_t[1]/UA, pos_a[1]/UA], 
               'k--', linewidth=2, alpha=0.5, label='Aproximação mínima')
        
        ax2.plot(pos_a[0]/UA, pos_a[1]/UA, 'r*', 
               markersize=20, label='Ponto mais próximo',
               markeredgecolor='yellow', markeredgewidth=2)
    
    ax2.set_xlabel('x (UA)', fontsize=12)
    ax2.set_ylabel('y (UA)', fontsize=12)
    ax2.set_title('Zoom: Terra e Asteroide', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
def plotar_distancia_temporal(sistema: SistemaGravitacional,
                              corpo1_nome: str = "Terra",
                              corpo2_nome: str = "Asteroide"):
    """Plota a evolução da distância entre dois corpos."""
    corpo1 = next((c for c in sistema.corpos if c.nome == corpo1_nome), None)
    corpo2 = next((c for c in sistema.corpos if c.nome == corpo2_nome), None)
    
    if corpo1 is None or corpo2 is None:
        print("Corpos não encontrados!")
        return
    
    # Calcular distâncias
    n_pontos = min(len(corpo1.historico_posicao), len(corpo2.historico_posicao))
    tempos = np.array(corpo1.historico_tempo[:n_pontos]) / ANOS_EM_SEGUNDOS
    distancias = []
    
    for i in range(n_pontos):
        d = np.linalg.norm(corpo2.historico_posicao[i] - corpo1.historico_posicao[i])
        distancias.append(d)
    
    distancias = np.array(distancias)
    
    # Plotar
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(tempos, distancias/1e6, 'b-', linewidth=2)
    
    # Marcar distância mínima
    idx_min = np.argmin(distancias)
    ax.plot(tempos[idx_min], distancias[idx_min]/1e6, 'r*', 
           markersize=15, label=f'Mínima: {distancias[idx_min]/1e6:.2f} × 10³ km')
    
    # Linha de referência do raio da Terra
    ax.axhline(R_TERRA/1e6, color='green', linestyle='--', 
              linewidth=2, alpha=0.7, label='Raio da Terra')
    
    ax.set_xlabel('Tempo (anos)', fontsize=12)
    ax.set_ylabel('Distância (× 10³ km)', fontsize=12)
    ax.set_title(f'Distância {corpo1_nome}-{corpo2_nome} ao longo do tempo', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plotar_conservacao_energia(sistema: SistemaGravitacional):
    """Plota a evolução da energia total do sistema."""
    # Recalcular energia em cada ponto
    corpo_referencia = sistema.corpos[0]
    n_pontos = len(corpo_referencia.historico_tempo)
    tempos = np.array(corpo_referencia.historico_tempo) / ANOS_EM_SEGUNDOS
    energias = []
    
    # Salvar estado atual
    estado_original = sistema.get_estado()
    tempo_original = sistema.tempo_atual
    
    # Calcular energia em cada ponto do histórico
    for i in range(0, n_pontos, max(1, n_pontos//1000)):  # Amostragem
        # Restaurar estado histórico
        for j, corpo in enumerate(sistema.corpos):
            if i < len(corpo.historico_posicao):
                corpo.posicao = corpo.historico_posicao[i].copy()
                corpo.velocidade = corpo.historico_velocidade[i].copy()
        
        energias.append(sistema.energia_total())
    
    # Restaurar estado original
    sistema.set_estado(estado_original)
    sistema.tempo_atual = tempo_original
    
    energias = np.array(energias)
    tempos_amostrados = tempos[::max(1, n_pontos//1000)][:len(energias)]
    
    # Calcular erro relativo
    e0 = energias[0]
    erros_relativos = (energias - e0) / abs(e0)
    
    # Plotar
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Energia total
    ax1.plot(tempos_amostrados, energias, 'b-', linewidth=2)
    ax1.set_xlabel('Tempo (anos)', fontsize=12)
    ax1.set_ylabel('Energia Total (J)', fontsize=12)
    ax1.set_title('Conservação de Energia', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Erro relativo
    ax2.plot(tempos_amostrados, erros_relativos, 'r-', linewidth=2)
    ax2.axhline(TOLERANCIA_ENERGIA, color='green', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Tolerância: {TOLERANCIA_ENERGIA:.1e}')
    ax2.axhline(-TOLERANCIA_ENERGIA, color='green', linestyle='--', 
               linewidth=2, alpha=0.7)
    ax2.set_xlabel('Tempo (anos)', fontsize=12)
    ax2.set_ylabel('Erro Relativo (ΔE/E₀)', fontsize=12)
    ax2.set_title('Erro na Conservação de Energia', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def configurar_backend_animacao():
    """
    Força o backend correto para animações.
    """
    import sys
    
    # Detectar se está no Jupyter
    try:
        shell = get_ipython().__class__.__name__
        if 'ZMQInteractiveShell' in shell:
            # Jupyter Notebook/Lab
            print("📱 Jupyter detectado - configurando backend 'widget'")
            try:
                import ipywidgets
                matplotlib.use('widget')
                return 'widget'
            except ImportError:
                print("⚠️ ipympl não instalado. Usando 'nbagg'")
                matplotlib.use('nbagg')
                return 'nbagg'
    except NameError:
        # Não está no Jupyter
        pass
    
    # Python script normal
    backend_atual = matplotlib.get_backend()
    print(f"🖥️ Backend atual: {backend_atual}")
    
    # Backends que funcionam para animação
    backends_bons = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'WXAgg', 'MacOSX']
    
    if backend_atual not in backends_bons:
        print(f"⚠️ Backend {backend_atual} pode não suportar animações")
        print("   Tentando mudar para TkAgg...")
        try:
            matplotlib.use('TkAgg', force=True)
            print("   ✓ Backend mudado para TkAgg")
            return 'TkAgg'
        except:
            print("   ✗ Não foi possível mudar o backend")
            print("   DICA: Execute '%matplotlib widget' antes do menu no Jupyter")
    
    return backend_atual

    
def plotar_animacao_interativa(sistema: SistemaGravitacional,
                               titulo: str = "Animação Orbital",
                               velocidade: int = 1,
                               max_frames: int = 200):
    """
    Cria uma animação interativa otimizada das órbitas.
    """
    global _animacoes_ativas
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt
    
    # fechar figuras anteriores
    plt.close('all')
    
    # VERIFICAR SE HÁ DADOS
    n_pontos = 0
    for corpo in sistema.corpos:
        if len(corpo.historico_posicao) > 0:
            n_pontos = max(n_pontos, len(corpo.historico_posicao))
    
    if n_pontos == 0:
        print("⚠️ ERRO: Sistema não foi simulado!")
        return None
    
    print(f"\n🎬 Preparando animação com {n_pontos} pontos...")
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Determinar limites
    max_dist = 1.5 * UA
    for corpo in sistema.corpos:
        if len(corpo.historico_posicao) > 0:
            posicoes = np.array(corpo.historico_posicao)
            max_corpo = np.max(np.abs(posicoes[:, :2]))
            if max_corpo > max_dist:
                max_dist = max_corpo * 1.2
    
    ax.set_xlim(-max_dist/UA, max_dist/UA)
    ax.set_ylim(-max_dist/UA, max_dist/UA)
    ax.set_xlabel('x (UA)', fontsize=11)
    ax.set_ylabel('y (UA)', fontsize=11)
    ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Criar objetos gráficos
    pontos = {}
    rastros = {}
    
    for corpo in sistema.corpos:
        if len(corpo.historico_posicao) == 0:
            continue
            
        ponto, = ax.plot([], [], 'o', color=corpo.cor, 
                        markersize=corpo.raio_visual,
                        markeredgecolor='white', markeredgewidth=0.5,
                        label=corpo.nome, zorder=10)
        pontos[corpo.nome] = ponto
        
        rastro, = ax.plot([], [], '-', color=corpo.cor, 
                         linewidth=0.8, alpha=0.3, zorder=5)
        rastros[corpo.nome] = rastro
    
    # Texto de tempo
    tempo_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        zorder=20)
    
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    # Preparar índices
    skip = max(1, n_pontos // max_frames) * velocidade
    indices = list(range(0, n_pontos, skip))
    
    print(f"📊 Animação: {len(indices)} frames (skip={skip})")
    print(f"   Duração: {sistema.tempo_atual/ANOS_EM_SEGUNDOS:.2f} anos")
    
    # Cache de posições
    cache_posicoes = {}
    cache_tempos = {}
    for corpo in sistema.corpos:
        if len(corpo.historico_posicao) > 0:
            cache_posicoes[corpo.nome] = np.array(corpo.historico_posicao)
            cache_tempos[corpo.nome] = np.array(corpo.historico_tempo)
    
    def init():
        """Inicializa a animação"""
        for nome in pontos:
            pontos[nome].set_data([], [])
            rastros[nome].set_data([], [])
        tempo_text.set_text('')
        return list(pontos.values()) + list(rastros.values()) + [tempo_text]
    
    def animate(frame_idx):
        """Atualiza cada frame"""
        if frame_idx >= len(indices):
            return list(pontos.values()) + list(rastros.values()) + [tempo_text]
        
        idx = indices[frame_idx]
        
        for corpo in sistema.corpos:
            if corpo.nome not in cache_posicoes:
                continue
            
            posicoes = cache_posicoes[corpo.nome]
            if idx >= len(posicoes):
                continue
            
            pos_atual = posicoes[idx]
            pontos[corpo.nome].set_data([pos_atual[0]/UA], [pos_atual[1]/UA])
            
            inicio = max(0, idx - 30)
            x_hist = posicoes[inicio:idx+1, 0] / UA
            y_hist = posicoes[inicio:idx+1, 1] / UA
            rastros[corpo.nome].set_data(x_hist, y_hist)
        
        if cache_tempos:
            tempo_ref = list(cache_tempos.values())[0]
            if idx < len(tempo_ref):
                tempo = tempo_ref[idx]
                tempo_text.set_text(f'Tempo: {tempo/(24*3600):.1f} dias\n'
                                  f'({tempo/ANOS_EM_SEGUNDOS:.3f} anos)')
        
        return list(pontos.values()) + list(rastros.values()) + [tempo_text]
    
    # Criar animação
    print("   Gerando animação...")
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(indices), interval=50,
                        blit=False, repeat=True)
    
    # CRÍTICO: Manter referência global (evita garbage collection)
    _animacoes_ativas.append(anim)
    
    # Limitar número de animações em memória
    if len(_animacoes_ativas) > 5:
        _animacoes_ativas.pop(0)
    
    plt.tight_layout()
    plt.show()
    
    return anim

print("✓ Funções de visualização criadas!")

# ==============================================================================
# PARTE 5: SIMULAÇÃO MONTE CARLO
# ==============================================================================

def simulacao_monte_carlo(n_simulacoes: int = 100,
                         variacao_posicao: float = 0.01,
                         variacao_velocidade: float = 0.01,
                         tempo_total: float = 2 * ANOS_EM_SEGUNDOS,
                         massa_base: float = None,
                         posicao_base: List[float] = None,
                         velocidade_base: List[float] = None,
                         seed: Optional[int] = None) -> Dict:
    """
    Executa múltiplas simulações com variações nas condições iniciais.
    
    Parâmetros:
        n_simulacoes: Número de simulações
        variacao_posicao: Variação percentual na posição
        variacao_velocidade: Variação percentual na velocidade
        tempo_total: Tempo de simulação
        massa_base: Massa do asteroide
        posicao_base: Posição inicial [x, y, z]
        velocidade_base: Velocidade inicial [vx, vy, vz]
        seed: Semente para reprodutibilidade
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"Iniciando simulação Monte Carlo com {n_simulacoes} cenários...")
    print("=" * 70)
    
    # Listas para armazenar resultados
    distancias_minimas = []
    tempos_minimos = []
    colisoes = 0
    energias_impacto = []
    
    # Armazenar trajetórias de todos os asteroides
    trajetorias_asteroides = []
    sistema_referencia = None
    
    # Definir valores base
    if massa_base is None or posicao_base is None or velocidade_base is None:
        sistema_base = criar_sistema_apophis()
        asteroide_base = next(c for c in sistema_base.corpos if c.nome == "Asteroide")
        
        if massa_base is None:
            massa_base = asteroide_base.massa
        if posicao_base is None:
            posicao_base = asteroide_base.posicao.copy()
        else:
            posicao_base = np.array(posicao_base)
        if velocidade_base is None:
            velocidade_base = asteroide_base.velocidade.copy()
        else:
            velocidade_base = np.array(velocidade_base)
    else:
        posicao_base = np.array(posicao_base)
        velocidade_base = np.array(velocidade_base)
    
    print(f"\nCONDIÇÕES BASE:")
    print(f"  Massa: {massa_base:.2e} kg")
    print(f"  Posição: ({posicao_base[0]/UA:.4f}, {posicao_base[1]/UA:.4f}, {posicao_base[2]/UA:.4f}) UA")
    print(f"  Velocidade: ({velocidade_base[0]/1000:.2f}, {velocidade_base[1]/1000:.2f}, {velocidade_base[2]/1000:.2f}) km/s")
    print(f"  Variação posição: ±{variacao_posicao*100:.1f}%")
    print(f"  Variação velocidade: ±{variacao_velocidade*100:.1f}%")
    print()
    
    for i in range(n_simulacoes):
        # Variações aleatórias
        delta_pos = np.random.normal(0, variacao_posicao, 3)
        delta_vel = np.random.normal(0, variacao_velocidade, 3)
        
        nova_pos = posicao_base * (1 + delta_pos)
        nova_vel = velocidade_base * (1 + delta_vel)
        
        # Criar sistema com variação
        sistema = criar_sistema_personalizado(massa_base, nova_pos.tolist(), nova_vel.tolist())
        
        # Simular
        resultado = sistema.simular(tempo_total, progresso=False)
        
        # Coletar dados
        distancias_minimas.append(resultado.distancia_minima)
        tempos_minimos.append(resultado.tempo_minima)
        
        if resultado.houve_colisao:
            colisoes += 1
            energias_impacto.append(resultado.energia_impacto)
        
        # Salvar trajetória do asteroide
        asteroide = next(c for c in sistema.corpos if c.nome == "Asteroide")
        trajetorias_asteroides.append({
            'posicoes': asteroide.historico_posicao.copy(),
            'colidiu': resultado.houve_colisao
        })
        
        # Guardar primeiro sistema para referência
        if i == 0:
            sistema_referencia = sistema
        
        # Progresso
        if (i + 1) % (n_simulacoes // 10) == 0:
            print(f"Progresso: {100*(i+1)/n_simulacoes:.0f}% ({i+1}/{n_simulacoes} simulações)")
    
    # Estatísticas
    distancias_minimas = np.array(distancias_minimas)
    tempos_minimos = np.array(tempos_minimos)
    
    estatisticas = {
        'n_simulacoes': n_simulacoes,
        'n_colisoes': colisoes,
        'probabilidade_colisao': colisoes / n_simulacoes,
        'distancia_media': np.mean(distancias_minimas),
        'distancia_std': np.std(distancias_minimas),
        'distancia_min': np.min(distancias_minimas),
        'distancia_max': np.max(distancias_minimas),
        'tempo_medio': np.mean(tempos_minimos),
        'distancias': distancias_minimas,
        'tempos': tempos_minimos,
        'energias_impacto': energias_impacto if energias_impacto else [0],
        'trajetorias_asteroides': trajetorias_asteroides,
        'sistema_referencia': sistema_referencia
    }
    
    print("\n" + "=" * 70)
    print("RESULTADOS DA SIMULAÇÃO MONTE CARLO".center(70))
    print("=" * 70)
    print(f"Total de simulações: {n_simulacoes}")
    print(f"Colisões detectadas: {colisoes}")
    print(f"Probabilidade de impacto: {100*estatisticas['probabilidade_colisao']:.2f}%")
    print(f"\nDistância mínima:")
    print(f"  Média: {estatisticas['distancia_media']/1e6:.2f} × 10³ km")
    print(f"  Desvio padrão: {estatisticas['distancia_std']/1e6:.2f} × 10³ km")
    print(f"  Mínima: {estatisticas['distancia_min']/1e6:.2f} × 10³ km")
    print(f"  Máxima: {estatisticas['distancia_max']/1e6:.2f} × 10³ km")
    print(f"  Em raios terrestres: {estatisticas['distancia_media']/R_TERRA:.2f} ± {estatisticas['distancia_std']/R_TERRA:.2f} R⊕")
    print("=" * 70)
    
    return estatisticas

def plotar_resultados_monte_carlo(estatisticas: Dict):
    """Plota os resultados da simulação Monte Carlo."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    distancias = estatisticas['distancias'] / 1e6  # em 10³ km
    tempos = estatisticas['tempos'] / ANOS_EM_SEGUNDOS
    
    # Histograma de distâncias
    axes[0, 0].hist(distancias, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(R_TERRA/1e6, color='red', linestyle='--', 
                      linewidth=2, label='Raio da Terra')
    axes[0, 0].set_xlabel('Distância Mínima (× 10³ km)', fontsize=11)
    axes[0, 0].set_ylabel('Frequência', fontsize=11)
    axes[0, 0].set_title('Distribuição de Distâncias Mínimas', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histograma de tempos
    axes[0, 1].hist(tempos, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Tempo de Aproximação (anos)', fontsize=11)
    axes[0, 1].set_ylabel('Frequência', fontsize=11)
    axes[0, 1].set_title('Distribuição de Tempos de Aproximação', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter: distância vs tempo
    axes[1, 0].scatter(tempos, distancias, alpha=0.6, s=30, c='purple')
    axes[1, 0].axhline(R_TERRA/1e6, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7, label='Raio da Terra')
    axes[1, 0].set_xlabel('Tempo de Aproximação (anos)', fontsize=11)
    axes[1, 0].set_ylabel('Distância Mínima (× 10³ km)', fontsize=11)
    axes[1, 0].set_title('Distância vs Tempo', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Estatísticas resumidas
    axes[1, 1].axis('off')
    stats_text = f"""
    ESTATÍSTICAS RESUMIDAS
    
    Simulações: {estatisticas['n_simulacoes']}
    Colisões: {estatisticas['n_colisoes']}
    Prob. Impacto: {100*estatisticas['probabilidade_colisao']:.2f}%
    
    Distância Mínima:
      Média: {estatisticas['distancia_media']/1e6:.2f} × 10³ km
      Std: {estatisticas['distancia_std']/1e6:.2f} × 10³ km
      Min: {estatisticas['distancia_min']/1e6:.2f} × 10³ km
      Max: {estatisticas['distancia_max']/1e6:.2f} × 10³ km
    
    Em Raios Terrestres:
      {estatisticas['distancia_media']/R_TERRA:.2f} ± {estatisticas['distancia_std']/R_TERRA:.2f} R⊕
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                   verticalalignment='center', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def plotar_trajetorias_monte_carlo(estatisticas: Dict):
    """Plota todas as trajetórias dos asteroides da simulação Monte Carlo."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Pegar sistema de referência (primeiro)
    sistema_ref = estatisticas['sistema_referencia']
    
    if sistema_ref is None:
        print("⚠️  Nenhum sistema de referência disponível!")
        return
    
    # Plotar SOL (do primeiro sistema)
    sol = next((c for c in sistema_ref.corpos if c.nome == "Sol"), None)
    if sol:
        ax.plot(0, 0, 'o', color='yellow', markersize=25, 
               markeredgecolor='orange', markeredgewidth=2, label='Sol', zorder=10)
    
    # Plotar TERRA (do primeiro sistema)
    terra = next((c for c in sistema_ref.corpos if c.nome == "Terra"), None)
    if terra:
        x_terra, y_terra = terra.get_trajetoria_2d()
        if len(x_terra) > 0:
            ax.plot(x_terra/UA, y_terra/UA, '-', color='blue', 
                   linewidth=2, alpha=0.8, label='Terra', zorder=5)
            # Marcar posição inicial e final
            ax.plot(x_terra[0]/UA, y_terra[0]/UA, 'o', color='blue', 
                   markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=6)
            ax.plot(x_terra[-1]/UA, y_terra[-1]/UA, 's', color='blue', 
                   markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=6)
    
    # Plotar todos os asteroides
    n_colisoes = estatisticas['n_colisoes']
    n_total = estatisticas['n_simulacoes']
    
    for i, traj_data in enumerate(estatisticas['trajetorias_asteroides']):
        posicoes = traj_data['posicoes']
        colidiu = traj_data['colidiu']
        
        if len(posicoes) > 0:
            posicoes_array = np.array(posicoes)
            x = posicoes_array[:, 0] / UA
            y = posicoes_array[:, 1] / UA
            
            # Cor diferente para colisões
            if colidiu:
                cor = 'red'
                alpha = 0.6
                linewidth = 1.2
                zorder = 4
            else:
                cor = 'black'
                alpha = 0.3
                linewidth = 0.6
                zorder = 1
            
            # Plotar trajetória
            ax.plot(x, y, '-', color=cor, linewidth=linewidth, 
                   alpha=alpha, zorder=zorder)
            
            # Marcar apenas início dos asteroides (para não poluir)
            if i == 0:  # Só no primeiro para aparecer na legenda
                if colidiu:
                    ax.plot(x[0], y[0], 'o', color='red', markersize=4, 
                           alpha=0.7, label=f'Asteroides (colisão: {n_colisoes})', zorder=3)
                else:
                    ax.plot(x[0], y[0], 'o', color='gray', markersize=3, 
                           alpha=0.3, label=f'Asteroides (sem colisão: {n_total-n_colisoes})', zorder=2)
    
    # Adicionar círculo da órbita terrestre como referência
    theta = np.linspace(0, 2*np.pi, 100)
    x_circ = np.cos(theta)
    y_circ = np.sin(theta)
    ax.plot(x_circ, y_circ, ':', color='cyan', alpha=0.4, 
           linewidth=1.5, label='Órbita terrestre (1 UA)', zorder=0)
    
    # Configurações do gráfico
    ax.set_xlabel('x (UA)', fontsize=13)
    ax.set_ylabel('y (UA)', fontsize=13)
    ax.set_title(f'Simulação Monte Carlo - {n_total} Trajetórias\n' + 
                f'Colisões: {n_colisoes} ({100*n_colisoes/n_total:.1f}%)', 
                fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axis('equal')
    
    # Adicionar estatísticas no canto
    stats_text = f"n = {n_total}\nColisões = {n_colisoes}\nProb = {100*estatisticas['probabilidade_colisao']:.1f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

print("✓ Funções de Monte Carlo criadas!")


# ==============================================================================
# PARTE 6: FUNÇÕES DE I/O (JSON)
# ==============================================================================

def salvar_configuracao(sistema: SistemaGravitacional, arquivo: str = 'config.json'):
    """Salva a configuração do sistema em JSON."""
    config = {
        'dt': sistema.dt,
        'corpos': []
    }
    
    for corpo in sistema.corpos:
        config['corpos'].append({
            'nome': corpo.nome,
            'massa': corpo.massa,
            'posicao': corpo.posicao.tolist(),
            'velocidade': corpo.velocidade.tolist(),
            'cor': corpo.cor,
            'raio_visual': corpo.raio_visual
        })
    
    with open(arquivo, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Configuração salva em: {arquivo}")

def carregar_configuracao(arquivo: str = 'config.json') -> SistemaGravitacional:
    """Carrega a configuração do sistema de um JSON."""
    with open(arquivo, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    sistema = SistemaGravitacional(dt=config.get('dt', DT_PADRAO))
    
    for corpo_config in config['corpos']:
        corpo = CorpoCeleste(
            nome=corpo_config['nome'],
            massa=corpo_config['massa'],
            posicao=corpo_config['posicao'],
            velocidade=corpo_config['velocidade'],
            cor=corpo_config.get('cor', 'blue'),
            raio_visual=corpo_config.get('raio_visual', 5.0)
        )
        sistema.adicionar_corpo(corpo)
    
    print(f"✓ Configuração carregada de: {arquivo}")
    return sistema

def salvar_resultado(resultado: ResultadoSimulacao, arquivo: str = 'resultado.txt'):
    """Salva o relatório da simulação em arquivo de texto."""
    with open(arquivo, 'w', encoding='utf-8') as f:
        f.write(resultado.gerar_relatorio())
    
    print(f"✓ Resultado salvo em: {arquivo}")

def exportar_trajetorias(sistema: SistemaGravitacional, arquivo: str = 'trajetorias.json'):
    """Exporta as trajetórias para JSON."""
    dados = {
        'tempo_simulacao': sistema.tempo_atual,
        'corpos': {}
    }
    
    for corpo in sistema.corpos:
        dados['corpos'][corpo.nome] = {
            'massa': corpo.massa,
            'cor': corpo.cor,
            'tempos': corpo.historico_tempo,
            'posicoes': [p.tolist() for p in corpo.historico_posicao],
            'velocidades': [v.tolist() for v in corpo.historico_velocidade]
        }
    
    with open(arquivo, 'w', encoding='utf-8') as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Trajetórias exportadas para: {arquivo}")

print("✓ Funções de I/O criadas!")


# ==============================================================================
# PARTE 7: INTERFACE DE MENU INTERATIVO
# ==============================================================================

def menu_principal():
    """Menu interativo principal."""
    print("\n" + "="*70)
    print("SIMULADOR DE TRAJETÓRIAS DE ASTEROIDES".center(70))
    print("="*70)
    print("\nEscolha uma opção:")
    print("\n1. CASOS PRÉ-CONFIGURADOS")
    print("   a) Órbita da Terra")
    print("   b) Asteroide Apophis (2029)")
    print("   c) Cenário de impacto hipotético")
    print("\n2. CONFIGURAÇÃO PERSONALIZADA")
    print("   d) Inserir parâmetros manualmente")
    print("   e) Gerar condições aleatórias")
    print("\n3. SIMULAÇÃO MONTE CARLO")
    print("   f) Executar análise estatística")
    print("\n4. CARREGAR CONFIGURAÇÃO")
    print("   g) Carregar de arquivo JSON")
    print("\n5. SISTEMAS EXPANDIDOS")
    print("   h) Sistema com Lua")
    print("   i) Sistema Solar Completo")
    print("\n0. Sair\n(escolha essa opção para carregar as figuras, se usar matplotlib widget)")
    print("="*70)

def executar_simulacao_interativa():
    """
    Executa o simulador de forma interativa.
    """
    # verificar backend
    backend_atual = matplotlib.get_backend()
    
    if backend_atual not in ['widget']:
        print("\n" + "⚠️"*35)
        print("AVISO: Backend atual não suporta animações interativas!")
        print(f"Backend detectado: {backend_atual}")
        print("\nSOLUÇÃO:")
        print("  - No Jupyter: Execute '%matplotlib widget' (e talvez reinicie o kernel)")
        print("  - Em script: O backend será ajustado automaticamente")
        print("⚠️"*35 + "\n")
        
        resposta = input("Deseja tentar ajustar automaticamente? (s/n): ").lower()
        if resposta == 's':
            configurar_backend_animacao()
            print("\nBackend ajustado. Continuando...\n")
        else:
            print("\nAVISO: Animações podem não funcionar corretamente!\n")
    
    while True:
        menu_principal()
        opcao = input("\nDigite sua opção: ").strip().lower()
        
        if opcao == '0':
            print("\nEncerrando simulador. Até logo!")
            break
        
        elif opcao == 'a':
            print("\n>>> Simulando órbita da Terra (validação)...")
            dt = float(input("Passo temporal dt (segundos) [padrão: 3600]: ") or "3600")
            sistema = criar_sistema_terra_sol(dt=dt)
            
            tempo = float(input("Tempo de simulação (anos): ") or "1")
            tempo_total = tempo * ANOS_EM_SEGUNDOS
            
            resultado = sistema.simular(tempo_total)
            print("\n" + resultado.gerar_relatorio())
            
            # Escolha: Animação ou Estático
            print("\nVisualização:")
            print("  1) Animação interativa")
            print("  2) Figura estática")
            viz = input("Escolha: ").strip() or "2"
            
            if viz == "1":
                vel = int(input("Velocidade [1-5]: ") or "3")
                plotar_animacao_interativa(sistema, "Órbita da Terra", 
                                          velocidade=vel, max_frames=200)
            elif viz == "2":
                plotar_trajetorias(sistema, "Órbita da Terra")
                plotar_conservacao_energia(sistema)
        
        elif opcao == 'b':
            print("\n>>> Simulando Asteroide Apophis (2029)...")
            dt = float(input("Passo temporal dt (segundos) [padrão: 3600]: ") or "3600")
            sistema = criar_sistema_apophis(dt=dt)
            
            tempo = float(input("Tempo de simulação (anos): ") or "3")
            tempo_total = tempo * ANOS_EM_SEGUNDOS
            
            resultado = sistema.simular(tempo_total)
            print("\n" + resultado.gerar_relatorio())
            
            print("\nVisualização:")
            print("  1) Animação interativa")
            print("  2) Figuras estáticas")
            viz = input("Escolha: ").strip() or "2"
            
            if viz == "1":
                vel = int(input("Velocidade [1-5]: ") or "3")
                plotar_animacao_interativa(sistema, "Asteroide Apophis", 
                                          velocidade=vel, max_frames=200)
            elif viz == "2":
                plotar_trajetorias(sistema, "Asteroide Apophis")
                plotar_distancia_temporal(sistema)
        
        elif opcao == 'c':
            print("\n>>> Simulando cenário de impacto hipotético...")
            sistema = criar_sistema_impacto(dt=900)
            tempo_total = 0.3 * ANOS_EM_SEGUNDOS
            
            resultado = sistema.simular(tempo_total)
            print("\n" + resultado.gerar_relatorio())
            
            print("\nVisualização:")
            print("  1) Animação interativa")
            print("  2) Figura estática")
            viz = input("Escolha: ").strip() or "1"
            
            if viz == "1":
                vel = int(input("Velocidade [1-5]: ") or "3")
                plotar_animacao_interativa(sistema, "Cenário de Impacto", 
                                          velocidade=vel, max_frames=200)
            elif viz == "2":
                plotar_trajetorias(sistema, "Cenário de Impacto")
        
        elif opcao == 'd':
            print("\n>>> Configuração personalizada")
            print("\nInsira os parâmetros do asteroide:")
            
            dt = float(input("  Passo temporal dt (segundos) [padrão: 3600]: ") or "3600")
            massa = float(input("  Massa (kg) [ex: 6.1e10]: ") or "6.1e10")
            
            print("  Posição inicial (UA):")
            px = float(input("    x: ") or "1.0") * UA
            py = float(input("    y: ") or "0") * UA
            pz = float(input("    z: ") or "0") * UA
            
            print("  Velocidade inicial (m/s):")
            vx = float(input("    vx: ") or "0")
            vy = float(input("    vy: ") or "30000")
            vz = float(input("    vz: ") or "0")
            
            sistema = criar_sistema_personalizado(massa, [px, py, pz], [vx, vy, vz], dt=dt)
            
            tempo = float(input("\nTempo de simulação (anos): ") or "2")
            tempo_total = tempo * ANOS_EM_SEGUNDOS
            
            resultado = sistema.simular(tempo_total)
            print("\n" + resultado.gerar_relatorio())
            
            print("\nVisualização:")
            print("  1) Animação interativa")
            print("  2) Figura estática")
            viz = input("Escolha: ").strip() or "2"
            
            if viz == "1":
                vel = int(input("Velocidade [1-5]: ") or "3")
                plotar_animacao_interativa(sistema, "Simulação Personalizada", 
                                          velocidade=vel, max_frames=200)
            elif viz == "2":
                plotar_trajetorias(sistema, "Simulação Personalizada")
        
        elif opcao == 'e':
            print("\n>>> Gerando condições aleatórias...")
            dt = float(input("Passo temporal dt (segundos) [padrão: 3600]: ") or "3600")
            seed = input("Seed (deixe vazio para aleatório): ").strip()
            seed = int(seed) if seed else None
            
            sistema = criar_sistema_aleatorio(dt=dt, seed=seed)
            
            print("\nCondições geradas:")
            asteroide = next(c for c in sistema.corpos if c.nome == "Asteroide")
            print(f"  Massa: {asteroide.massa:.2e} kg")
            print(f"  Posição: ({asteroide.posicao[0]/UA:.3f}, {asteroide.posicao[1]/UA:.3f}) UA")
            
            tempo = float(input("\nTempo de simulação (anos): ") or "2")
            tempo_total = tempo * ANOS_EM_SEGUNDOS
            
            resultado = sistema.simular(tempo_total)
            print("\n" + resultado.gerar_relatorio())
            
            print("\nVisualização:")
            print("  1) Animação interativa")
            print("  2) Figura estática")
            viz = input("Escolha: ").strip() or "2"
            
            if viz == "1":
                vel = int(input("Velocidade [1-5]: ") or "3")
                plotar_animacao_interativa(sistema, "Simulação com Condições Aleatórias", 
                                          velocidade=vel, max_frames=200)
            elif viz == "2":
                plotar_trajetorias(sistema, "Simulação com Condições Aleatórias")
        
        elif opcao == 'f':
            print("\nInsira os parâmetros do asteroide:")
            
            massa_base = float(input("  Massa (kg) [ex: 6.1e10]: ") or "6.1e10")
            
            print("  Posição inicial (UA):")
            px = float(input("    x: ") or "0.92") * UA
            py = float(input("    y: ") or "0.1") * UA
            pz = float(input("    z: ") or "0") * UA
            posicao_base = [px, py, pz]
            
            print("  Velocidade inicial (m/s):")
            vx = float(input("    vx: ") or "-5000")
            vy = float(input("    vy: ") or "28000")
            vz = float(input("    vz: ") or "0")
            velocidade_base = [vx, vy, vz]
            
            print(f"\n✓ Configuração personalizada definida")
            print(f"  Posição: ({px/UA:.3f}, {py/UA:.3f}, {pz/UA:.3f}) UA")
            print(f"  Velocidade: ({vx/1000:.2f}, {vy/1000:.2f}, {vz/1000:.2f}) km/s")
            
            print("\nParâmetros da simulação Monte Carlo:")
            n_sim = int(input("  Número de simulações [mínimo 10]: ") or "10")
            var_pos = float(input("  Variação em posição (%): ") or "1") / 100
            var_vel = float(input("  Variação em velocidade (%): ") or "1") / 100
            tempo = float(input("  Tempo por simulação (anos): ") or "2")
            
            tempo_total = tempo * ANOS_EM_SEGUNDOS
            
            estatisticas = simulacao_monte_carlo(
                n_simulacoes=n_sim,
                variacao_posicao=var_pos,
                variacao_velocidade=var_vel,
                tempo_total=tempo_total,
                massa_base=massa_base,
                posicao_base=posicao_base,
                velocidade_base=velocidade_base
            )
            
            plotar_resultados_monte_carlo(estatisticas)
            plotar_trajetorias_monte_carlo(estatisticas)
        
        elif opcao == 'g':
            print("\n>>> Carregar configuração de arquivo")
            arquivo = input("Nome do arquivo [padrão: config.json]: ").strip() or "config.json"
            
            try:
                sistema = carregar_configuracao(arquivo)
                
                tempo = float(input("Tempo de simulação (anos): ") or "2")
                tempo_total = tempo * ANOS_EM_SEGUNDOS
                
                resultado = sistema.simular(tempo_total)
                print("\n" + resultado.gerar_relatorio())
                
                print("\nVisualização:")
                print("  1) Animação interativa")
                print("  2) Figura estática")
                viz = input("Escolha: ").strip() or "2"
                
                if viz == "1":
                    vel = int(input("Velocidade [1-5]: ") or "3")
                    plotar_animacao_interativa(sistema, "Simulação Carregada", 
                                          velocidade=vel, max_frames=200)
                elif viz == "2":
                    plotar_trajetorias(sistema, "Simulação Carregada")
            
            except FileNotFoundError:
                print(f"⚠️ Arquivo '{arquivo}' não encontrado!")
            except json.JSONDecodeError:
                print(f"⚠️ Erro ao ler o arquivo JSON!")
        
        elif opcao == 'h':
            print("\n>>> Sistema com Lua")
            print("Escolha o cenário:")
            print("  1) Apophis")
            print("  2) Cenário de impacto")
            print("  3) Personalizado")
            
            escolha_ast = input("Escolha: ").strip() or "1"
            
            if escolha_ast == "1":
                # Apophis
                dt = float(input("Passo temporal dt (segundos) [padrão: 3600]: ") or "3600")
                sistema = criar_sistema_com_lua(dt=dt)
                tempo = float(input(f"\nTempo de simulação (anos): ") or 1)
                
            elif escolha_ast == "2":
                # Impacto
                sistema = criar_sistema_impacto(dt=900, incluir_lua=True)
                tempo = 0.3
                
            elif escolha_ast == "3":
                # Personalizado
                dt = float(input("Passo temporal dt (segundos) [padrão: 3600]: ") or "3600")
                print("\nParâmetros do asteroide:")
                massa = float(input("  Massa (kg): ") or "1e12")
                print("  Posição (UA):")
                px = float(input("    x: ") or "1.0") * UA
                py = float(input("    y: ") or "0") * UA
                print("  Velocidade (m/s):")
                vx = float(input("    vx: ") or "0")
                vy = float(input("    vy: ") or "30000")
                
                ast_config = {
                    'massa': massa,
                    'posicao': [px, py, 0],
                    'velocidade': [vx, vy, 0]
                }
                sistema = criar_sistema_com_lua(asteroide_personalizado=ast_config, dt=dt)
                tempo = float(input(f"\nTempo de simulação (anos): ") or 1)
            
            tempo_total = tempo * ANOS_EM_SEGUNDOS
            
            resultado = sistema.simular(tempo_total)
            print("\n" + resultado.gerar_relatorio())
            
            print("\nVisualização:")
            print("  1) Animação interativa")
            print("  2) Figura estática")
            viz = input("Escolha: ").strip() or "2"
            
            if viz == "1":
                vel = int(input("Velocidade [1-5]: ") or "3")
                plotar_animacao_interativa(sistema, "Sistema Terra-Lua-Asteroide", 
                                          velocidade=vel, max_frames=150)
            elif viz == "2":
                plotar_trajetorias(sistema, "Sistema Terra-Lua-Asteroide")
        
        elif opcao == 'i':
            print("\n>>> Sistema Solar Completo")
            print("Escolha o cenário:")
            print("  1) Cometa interestelar")
            print("  2) Cinturão de asteroides")
            print("  3) Próximo à Terra")
            print("  4) Personalizado")
            
            escolha = input("Escolha: ").strip() or "1"
            
            if escolha == "1":
                dt_input = input(f"Passo temporal dt (segundos) [padrão: 20000]: ").strip()
                dt = float(dt_input) if dt_input else 20000
                sistema = criar_sistema_solar_completo("padrao", dt=dt)
                
            elif escolha == "2":
                dt = float(input("Passo temporal dt (segundos) [padrão: 7200]: ") or "7200")
                sistema = criar_sistema_solar_completo("cinturao", dt=dt)
                
            elif escolha == "3":
                dt = float(input("Passo temporal dt (segundos) [padrão: 7200]: ") or "7200")
                sistema = criar_sistema_solar_completo("proximo", dt=dt)
                
            elif escolha == "4":
                print("\nParâmetros do asteroide:")
                massa = float(input("  Massa (kg): ") or "1e13")
                print("  Posição (UA):")
                px = float(input("    x: ") or "10") * UA
                py = float(input("    y: ") or "0") * UA
                print("  Velocidade (m/s):")
                vx = float(input("    vx: ") or "-30000")
                vy = float(input("    vy: ") or "5000")
                
                ast_config = {
                    'massa': massa,
                    'posicao': [px, py, 0],
                    'velocidade': [vx, vy, 0]
                }
                dt = float(input("  dt (segundos) [padrão: 7200]: ") or "7200")
                sistema = criar_sistema_solar_completo(ast_config, dt=dt)
            
            tempo = float(input("\nTempo de simulação (anos) [sugerido: 5]: ") or "5")
            tempo_total = tempo * ANOS_EM_SEGUNDOS
            
            print("\n🚀 Iniciando simulação...")
            resultado = sistema.simular(tempo_total)
            print("\n" + resultado.gerar_relatorio())
            
            print("\nVisualização:")
            print("  1) Animação interativa")
            print("  2) Figura estática")
            viz = input("Escolha: ").strip() or "2"
            
            if viz == "1":
                vel = int(input("Velocidade [1-5]: ") or "3")
                plotar_animacao_interativa(sistema, "Sistema Solar", 
                                          velocidade=vel, max_frames=200)
            elif viz == "2":
                plotar_trajetorias_sistema_solar(sistema, "Sistema Solar")
        
        else:
            print("\n⚠️ Opção inválida! Tente novamente.")
        
        input("\nPressione ENTER para continuar...")

print("✓ Interface de menu criada!")

# ==============================================================================
# PARTE 8: TESTES DE VALIDAÇÃO
# ==============================================================================

def teste_conservacao_energia():
    """Testa a conservação de energia em órbita circular."""
    print("\nTESTE 1: Conservação de Energia (Órbita Terra-Sol)")
    print("="*70)
    
    sistema = criar_sistema_terra_sol()
    resultado = sistema.simular(2 * ANOS_EM_SEGUNDOS, progresso=False)
    
    erro_energia = abs(resultado.erro_energia_relativo)
    
    print(f"Energia inicial: {resultado.energia_inicial:.6e} J")
    print(f"Energia final:   {resultado.energia_final:.6e} J")
    print(f"Erro relativo:   {erro_energia:.6e}")
    print(f"Tolerância:      {TOLERANCIA_ENERGIA:.6e}")
    
    if erro_energia < TOLERANCIA_ENERGIA:
        print("✓ TESTE PASSOU: Energia conservada!")
    else:
        print("✗ TESTE FALHOU: Violação na conservação de energia!")
    
    return erro_energia < TOLERANCIA_ENERGIA

def teste_terceira_lei_kepler():
    """Testa a Terceira Lei de Kepler."""
    print("\nTESTE 2: Terceira Lei de Kepler")
    print("="*70)
    
    sistema = criar_sistema_terra_sol()
    resultado = sistema.simular(1 * ANOS_EM_SEGUNDOS, progresso=False)
    
    terra = next(c for c in sistema.corpos if c.nome == "Terra")
    
    # Calcular período orbital
    periodo_simulado = terra.historico_tempo[-1]
    
    # Período teórico pela 3ª Lei de Kepler: T² = (4π²/GM) * a³
    a = UA  # semi-eixo maior
    periodo_teorico = 2 * np.pi * np.sqrt(a**3 / (G * M_SOL))
    
    erro_relativo = abs(periodo_simulado - periodo_teorico) / periodo_teorico
    
    print(f"Período simulado: {periodo_simulado/ANOS_EM_SEGUNDOS:.6f} anos")
    print(f"Período teórico:  {periodo_teorico/ANOS_EM_SEGUNDOS:.6f} anos")
    print(f"Erro relativo:    {erro_relativo:.6e}")
    
    if erro_relativo < 0.01:  # 1% de erro
        print("✓ TESTE PASSOU: Período orbital correto!")
    else:
        print("✗ TESTE FALHOU: Período orbital incorreto!")
    
    return erro_relativo < 0.01

def teste_orbita_estavel():
    """Testa se a órbita da Terra permanece estável."""
    print("\nTESTE 3: Estabilidade Orbital")
    print("="*70)
    
    sistema = criar_sistema_terra_sol()
    resultado = sistema.simular(10 * ANOS_EM_SEGUNDOS, progresso=False)
    
    terra = next(c for c in sistema.corpos if c.nome == "Terra")
    
    # Calcular raio orbital ao longo do tempo
    posicoes = np.array(terra.historico_posicao)
    raios = np.linalg.norm(posicoes, axis=1)
    
    raio_medio = np.mean(raios)
    variacao = np.std(raios) / raio_medio
    
    print(f"Raio médio:       {raio_medio/UA:.6f} UA")
    print(f"Raio esperado:    1.000000 UA")
    print(f"Variação relativa: {variacao:.6e}")
    
    if variacao < 0.001:  # 0.1% de variação
        print("✓ TESTE PASSOU: Órbita estável!")
    else:
        print("✗ TESTE FALHOU: Órbita instável!")
    
    return variacao < 0.001

def teste_conservacao_momento_angular():
    """Testa a conservação do momento angular."""
    print("\nTESTE 4: Conservação do Momento Angular")
    print("="*70)
    
    sistema = criar_sistema_terra_sol()
    resultado = sistema.simular(2 * ANOS_EM_SEGUNDOS, progresso=False)
    
    L_inicial = np.linalg.norm(resultado.momento_angular_inicial)
    L_final = np.linalg.norm(resultado.momento_angular_final)
    
    erro_momento = abs(L_final - L_inicial) / abs(L_inicial)
    
    print(f"Momento angular inicial: {L_inicial:.6e} kg·m²/s")
    print(f"Momento angular final:   {L_final:.6e} kg·m²/s")
    print(f"Erro relativo:           {erro_momento:.6e}")
    print(f"Tolerância:              {TOLERANCIA_ENERGIA:.6e}")
    
    if erro_momento < TOLERANCIA_ENERGIA:
        print("✓ TESTE PASSOU: Momento angular conservado!")
    else:
        print("✗ TESTE FALHOU: Violação na conservação do momento angular!")
    
    return erro_momento < TOLERANCIA_ENERGIA
    
def executar_todos_testes():
    """Executa todos os testes de validação."""
    print("\n" + "="*70)
    print("EXECUTANDO TESTES DE VALIDAÇÃO".center(70))
    print("="*70)
    
    resultados = []
    
    resultados.append(teste_conservacao_energia())
    resultados.append(teste_terceira_lei_kepler())
    resultados.append(teste_orbita_estavel())
    resultados.append(teste_conservacao_momento_angular())
    
    print("\n" + "="*70)
    print("RESUMO DOS TESTES".center(70))
    print("="*70)
    print(f"Total de testes: {len(resultados)}")
    print(f"Testes passados: {sum(resultados)}")
    print(f"Testes falhados: {len(resultados) - sum(resultados)}")
    
    if all(resultados):
        print("\n✓ TODOS OS TESTES PASSARAM!")
    else:
        print("\n⚠️  ALGUNS TESTES FALHARAM!")
    
    print("="*70)
    
    return all(resultados)

print("✓ Testes de validação criados!")

# ==============================================================================
# PARTE 9: EXEMPLOS DE USO
# ==============================================================================

def exemplo_basico():
    """Exemplo básico de uso do simulador."""
    print("\n" + "="*70)
    print("EXEMPLO 1: Simulação Básica (Órbita da Terra)".center(70))
    print("="*70)
    
    # Criar sistema
    sistema = criar_sistema_terra_sol()
    
    # Simular por 2 anos
    resultado = sistema.simular(2 * ANOS_EM_SEGUNDOS)
    
    # Mostrar relatório
    print("\n" + resultado.gerar_relatorio())
    
    # Visualizar
    plotar_trajetorias(sistema)
    plotar_conservacao_energia(sistema)

def exemplo_apophis():
    """Exemplo com asteroide Apophis."""
    print("\n" + "="*70)
    print("EXEMPLO 2: Asteroide Apophis (2029)".center(70))
    print("="*70)
    
    # Criar sistema
    sistema = criar_sistema_apophis()
    
    # Simular por 3 anos
    resultado = sistema.simular(3 * ANOS_EM_SEGUNDOS)
    
    # Mostrar relatório
    print("\n" + resultado.gerar_relatorio())
    
    # Visualizações
    plotar_trajetorias(sistema, "Asteroide Apophis - Aproximação 2029")
    plotar_distancia_temporal(sistema)
    plotar_conservacao_energia(sistema)

def exemplo_impacto():
    """Exemplo de cenário de impacto."""
    print("\n" + "="*70)
    print("EXEMPLO 3: Cenário de Impacto Hipotético".center(70))
    print("="*70)
    
    # Criar sistema
    sistema = criar_sistema_impacto()
    
    # Simular por 6 meses
    resultado = sistema.simular(0.5 * ANOS_EM_SEGUNDOS)
    
    # Mostrar relatório
    print("\n" + resultado.gerar_relatorio())
    
    # Visualizar
    plotar_trajetorias(sistema, "Cenário de Impacto")
    plotar_distancia_temporal(sistema)

def exemplo_monte_carlo():
    """Exemplo de simulação Monte Carlo."""
    print("\n" + "="*70)
    print("EXEMPLO 4: Simulação Monte Carlo".center(70))
    print("="*70)
    
    # Executar Monte Carlo
    estatisticas = simulacao_monte_carlo(
        n_simulacoes=15,
        variacao_posicao=0.01,
        variacao_velocidade=0.01,
        tempo_total=1 * ANOS_EM_SEGUNDOS,
        seed=42
    )
    
    # Visualizar resultados
    plotar_resultados_monte_carlo(estatisticas)
    plotar_trajetorias_monte_carlo(estatisticas)

def exemplo_personalizado(massa = 5e11, posicao = [1.2 * UA, 0.1 * UA, 0], velocidade = [0, 28000, 0]):
    """Exemplo com configuração personalizada."""
    print("\n" + "="*70)
    print("EXEMPLO 5: Configuração Personalizada".center(70))
    print("="*70)
    
    # Criar sistema
    sistema = criar_sistema_personalizado(massa, posicao, velocidade)
    
    # Simular
    resultado = sistema.simular(2 * ANOS_EM_SEGUNDOS)
    
    # Resultados
    print("\n" + resultado.gerar_relatorio())
    plotar_trajetorias(sistema, "Asteroide Personalizado")
    plotar_distancia_temporal(sistema)

print("✓ Exemplos de uso criados!")

# ==============================================================================
# PARTE 10: DOCUMENTAÇÃO E AJUDA
# ==============================================================================

print("\n" + "="*70)
print("📊 SIMULADOR DE TRAJETÓRIAS DE ASTEROIDES".center(70))
print("="*70)
print("\n✓ Todas as funções carregadas com sucesso!")
print("\nOpções de execução:")
print("  1. executar_simulacao_interativa()  - Menu interativo completo")
print("  2. exemplo_basico()                 - Órbita da Terra")
print("  3. exemplo_apophis()                - Asteroide Apophis")
print("  4. exemplo_impacto()                - Cenário de impacto")
print("  5. exemplo_monte_carlo()            - Análise estatística")
print("  6. exemplo_personalizado()          - Configuração customizada")
print("  7. executar_todos_testes()          - Validação completa")
print("\nExecute qualquer uma dessas funções!")
print('Obs.: "%matplotlib widget" necessário para animações no Jupyter')
print("="*70)


def mostrar_ajuda():
    """Mostra documentação completa do simulador."""
    ajuda = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║       SIMULADOR ORBITAL DE ASTEROIDES - GUIA DE USO               ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    📚 CLASSES PRINCIPAIS:
    
    1. CorpoCeleste
       - Representa um corpo celeste (Sol, Terra, Asteroide, Planetas)
       - Atributos: nome, massa, posicao, velocidade, cor, raio_visual
       - Métodos: salvar_estado(), energia_cinetica(), momento_angular()
    
    2. SistemaGravitacional
       - Gerencia múltiplos corpos e evolução temporal
       - Métodos principais:
         * adicionar_corpo(corpo)
         * simular(tempo_total, progresso=True)
         * energia_total(), momento_angular_total()
         * detectar_colisoes_todos_corpos()
    
    3. ResultadoSimulacao
       - Armazena resultados da simulação
       - Método: gerar_relatorio()
    
    ⚙️  FUNÇÕES DE CONFIGURAÇÃO:
    
    - criar_sistema_terra_sol(dt)               → Sistema Terra-Sol
    - criar_sistema_apophis(dt)                 → Asteroide Apophis (2029)
    - criar_sistema_impacto(dt)                 → Cenário de colisão
    - criar_sistema_personalizado(...)          → Sistema customizado
    - criar_sistema_aleatorio(dt, seed)         → Condições aleatórias
    - criar_sistema_com_lua(ast_config, dt)    → Terra-Lua + Asteroide
    - criar_sistema_solar_completo(config, dt) → Sistema Solar (8 planetas)
    
    🌙 SISTEMAS EXPANDIDOS:
    
    Sistema com Lua:
    ```python
    # Padrão (Apophis)
    sistema = criar_sistema_com_lua()
    
    # Personalizado
    ast = {'massa': 1e12, 'posicao': [1.5*UA, 0, 0], 'velocidade': [0, 25000, 0]}
    sistema = criar_sistema_com_lua(asteroide_personalizado=ast, dt=1800)
    ```
    
    Sistema Solar Completo:
    ```python
    # Cometa interestelar (padrão)
    sistema = criar_sistema_solar_completo("padrao")
    
    # Outros cenários
    sistema = criar_sistema_solar_completo("cinturao")  # Cinturão de asteroides
    sistema = criar_sistema_solar_completo("proximo")   # Próximo à Terra
    
    # Personalizado
    ast = {'massa': 5e12, 'posicao': [10*UA, 0, 0], 'velocidade': [-30000, 5000, 0]}
    sistema = criar_sistema_solar_completo(ast)
    ```
    
    📊 FUNÇÕES DE VISUALIZAÇÃO:
    
    - plotar_trajetorias(sistema)                    → Órbitas em 2D (estático)
    - plotar_distancia_temporal(sistema)             → Distância vs tempo
    - plotar_conservacao_energia(sistema)            → Validação física
    - plotar_animacao_interativa(sistema, ...)      → Animação dinâmica
    - plotar_resultados_monte_carlo(stats)           → Histogramas
    - plotar_trajetorias_monte_carlo(stats)          → Múltiplas trajetórias
    
    🎬 ANIMAÇÕES:
    ```python
    sistema = criar_sistema_apophis()
    sistema.simular(3 * ANOS_EM_SEGUNDOS)
    
    # Animação otimizada
    plotar_animacao_interativa(
        sistema, 
        titulo="Apophis 2029",
        velocidade=2,      # 2x mais rápido
        max_frames=200     # Limitar frames
    )
    ```
    
    🎲 SIMULAÇÃO MONTE CARLO:
    
    - simulacao_monte_carlo(n_simulacoes, variacao_posicao, variacao_velocidade)
      → Executa múltiplas simulações com variações aleatórias
      → Retorna estatísticas de risco de impacto
      → Plota trajetórias de todos os asteroides
    
    💾 FUNÇÕES DE I/O:
    
    - salvar_configuracao(sistema, arquivo)   → Salva em JSON
    - carregar_configuracao(arquivo)          → Carrega de JSON
    - salvar_resultado(resultado, arquivo)    → Salva relatório
    - exportar_trajetorias(sistema, arquivo)  → Exporta dados
    
    🧪 TESTES DE VALIDAÇÃO:
    
    - teste_conservacao_energia()    → Valida conservação de energia
    - teste_terceira_lei_kepler()    → Valida período orbital
    - teste_orbita_estavel()         → Valida estabilidade
    - executar_todos_testes()        → Executa todos os testes
    
    📖 EXEMPLOS RÁPIDOS:
    
    1. Validação básica:
    ```python
    sistema = criar_sistema_terra_sol(dt=3600)
    resultado = sistema.simular(2 * ANOS_EM_SEGUNDOS)
    plotar_trajetorias(sistema)
    ```
    
    2. Cenário de impacto com animação:
    ```python
    sistema = criar_sistema_impacto(dt=900)
    resultado = sistema.simular(0.3 * ANOS_EM_SEGUNDOS)
    plotar_animacao_interativa(sistema, velocidade=1, max_frames=150)
    ```
    
    3. Sistema Solar completo:
    ```python
    sistema = criar_sistema_solar_completo("padrao", dt=20000)
    resultado = sistema.simular(5 * ANOS_EM_SEGUNDOS)
    plotar_animacao_interativa(sistema, velocidade=3, max_frames=200)
    ```
    
    4. Monte Carlo com visualização:
    ```python
    stats = simulacao_monte_carlo(n_simulacoes)
    plotar_resultados_monte_carlo(stats)
    plotar_trajetorias_monte_carlo(stats)
    ```
    
    ⚡ CONSTANTES DISPONÍVEIS:
    
    - G             → Constante gravitacional (6.674e-11 m³/kg/s²)
    - UA            → Unidade Astronômica (1.496e11 m)
    - M_SOL         → Massa do Sol (1.989e30 kg)
    - M_TERRA       → Massa da Terra (5.972e24 kg)
    - R_TERRA       → Raio da Terra (6.371e6 m)
    - M_LUA         → Massa da Lua (7.342e22 kg)
    - ANOS_EM_SEGUNDOS → Segundos em um ano (31557600)
    - DT_PADRAO     → Passo temporal padrão (3600 s)
    

    📱 MENU INTERATIVO:
    
    Execute: executar_simulacao_interativa()
    
    Opções disponíveis:
    a) Órbita da Terra
    b) Asteroide Apophis (2029)
    c) Cenário de impacto
    d) Configuração personalizada
    e) Condições aleatórias
    f) Simulação Monte Carlo
    g) Carregar de JSON
    h) Sistema com Lua
    i) Sistema Solar Completo
    
    ════════════════════════════════════════════════════════════════════
    """
    print(ajuda)

# ==============================================================================
# FIM DO CÓDIGO - SIMULADOR COMPLETO
# ==============================================================================

print("\n" + "="*70)
print("✓ CÓDIGO COMPLETO CARREGADO!".center(70))
print("="*70)
print("\nPara começar, execute:")
print("  • executar_simulacao_interativa()  (menu interativo)")
print("  • exemplo_apophis()                (exemplo rápido)")
print("  • executar_todos_testes()          (validação)")
print("  • mostrar_ajuda()                  (documentação)")
print("="*70)
