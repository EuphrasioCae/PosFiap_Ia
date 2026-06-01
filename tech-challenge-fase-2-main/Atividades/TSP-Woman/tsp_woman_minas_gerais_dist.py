# -*- coding: utf-8 -*-
"""
Sistema de Roteamento Médico para Saúde da Mulher - Minas Gerais
COM ALTA PRIORIZAÇÃO DE ATENDIMENTOS URGENTES
"""

import pygame
from pygame.locals import *
import random
import math
import copy
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Set
from enum import Enum
import sys
from collections import deque

# Configurar encoding para Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# ==================== CONSTANTES ====================

WIDTH, HEIGHT = 1400, 900
NODE_RADIUS = 12
FPS = 60

# Dimensões dos painéis
MAP_WIDTH = WIDTH - 450
INFO_PANEL_WIDTH = 250
CONTROL_PANEL_WIDTH = 200
INFO_START_X = MAP_WIDTH
CONTROL_START_X = MAP_WIDTH + INFO_PANEL_WIDTH

# Limites geográficos de Minas Gerais
MG_LAT_MIN = -22.8
MG_LAT_MAX = -14.2
MG_LON_MIN = -51.5
MG_LON_MAX = -39.5

# Cores
WHITE = (255, 255, 255)
BLACK = (10, 10, 20)
DARK_GRAY = (30, 30, 40)
MEDIUM_GRAY = (60, 60, 70)
LIGHT_GRAY = (200, 200, 210)

# Cores de prioridade
COLOR_EMERGENCY = (255, 30, 60)
COLOR_HIGH = (255, 120, 0)
COLOR_MEDIUM = (255, 220, 0)
COLOR_LOW = (50, 220, 50)
COLOR_ROUTINE = (100, 100, 120)

# Cores da interface
PRIMARY_COLOR = (0, 150, 200)
SECONDARY_COLOR = (200, 50, 50)
SUCCESS_COLOR = (50, 200, 100)
WARNING_COLOR = (255, 180, 50)
CARD_BG = (25, 25, 35)
HOVER_COLOR = (0, 180, 230)
PANEL_BG = (20, 20, 30)

# Pesos para o cálculo de fitness (ajustados para dar MUITO peso à prioridade)
# QUANTO MENOR O FITNESS, MELHOR A SOLUÇÃO
FITNESS_WEIGHTS = {
    'distance_weight': 0.3,      # Peso da distância (menor importância)
    'priority_weight': 5.0,      # Peso da prioridade (ALTÍSSIMA importância)
    'urgency_penalty': 10.0,     # Penalidade para emergências no fim da rota
    'position_penalty': 8.0,     # Penalidade de posição para alta prioridade
    'time_window_weight': 2.0    # Peso para janelas de tempo
}

# ==================== ENUMS ====================

class UrgencyLevel(Enum):
    EMERGENCY = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    ROUTINE = 1
    
    @property
    def display_name(self):
        names = {5: "EMERGENCIA", 4: "ALTA", 3: "MEDIA", 2: "BAIXA", 1: "ROTINA"}
        return names[self.value]
    
    @property
    def icon(self):
        icons = {5: "🚨", 4: "⚠️", 3: "📌", 2: "📍", 1: "●"}
        return icons[self.value]

class MedicalCategory(Enum):
    PREGNANCY = 1
    POSTPARTUM = 2
    HORMONAL = 3
    EMERGENCY = 4
    PRENATAL = 5
    DOMESTIC_VIOLENCE = 6
    REMOTE_AREA = 7
    ROUTINE_CHECKUP = 8
    
    @property
    def display_name(self):
        names = {
            1: "Gestacao", 2: "Pos-parto", 3: "Hormonal", 4: "Emergencia",
            5: "Pre-natal", 6: "Violencia", 7: "Area Remota", 8: "Check-up"
        }
        return names[self.value]
    
    @property
    def icon(self):
        icons = {1: "🤰", 2: "👶", 3: "💊", 4: "🚨", 5: "👩‍⚕️", 6: "🛡️", 7: "🏔️", 8: "📋"}
        return icons[self.value]

class InitializationMethod(Enum):
    RANDOM = "Aleatorio"
    PRIORITY = "Prioridade Pura"
    NEAREST_NEIGHBOR = "Nearest Neighbor"
    HYBRID = "Hibrido"
    CLUSTER = "Cluster por Regiao"

class SelectionMethod(Enum):
    TOURNAMENT = "Torneio"
    ELITIST_TOURNAMENT = "Torneio Elitista"
    ROULETTE = "Roleta"
    RANK = "Ranking"
    BOLTZMANN = "Boltzmann"

# ==================== CIDADES DE MINAS GERAIS ====================

CITIES_MG = {
    "Montes Claros": {"lat": -16.7333, "lon": -43.8667, "region": "Norte"},
    "Janaúba": {"lat": -15.8000, "lon": -43.3000, "region": "Norte"},
    "Salinas": {"lat": -16.1667, "lon": -42.2833, "region": "Norte"},
    "Porteirinha": {"lat": -15.7500, "lon": -43.0333, "region": "Norte"},
    "Manga": {"lat": -14.7500, "lon": -44.0000, "region": "Norte"},
    "Poços de Caldas": {"lat": -21.7833, "lon": -46.5667, "region": "Sul"},
    "Pouso Alegre": {"lat": -22.2333, "lon": -45.9333, "region": "Sul"},
    "Varginha": {"lat": -21.5500, "lon": -45.4333, "region": "Sul"},
    "Itajubá": {"lat": -22.4167, "lon": -45.4500, "region": "Sul"},
    "São Lourenço": {"lat": -22.1167, "lon": -45.0500, "region": "Sul"},
    "Caxambu": {"lat": -21.9833, "lon": -44.9333, "region": "Sul"},
    "Uberlândia": {"lat": -18.9167, "lon": -48.2833, "region": "Triangulo"},
    "Uberaba": {"lat": -19.7333, "lon": -47.9167, "region": "Triangulo"},
    "Araguari": {"lat": -18.6500, "lon": -48.2000, "region": "Triangulo"},
    "Ituiutaba": {"lat": -18.9667, "lon": -49.4667, "region": "Triangulo"},
    "Frutal": {"lat": -20.0167, "lon": -48.9333, "region": "Triangulo"},
    "Iturama": {"lat": -19.7333, "lon": -50.2000, "region": "Triangulo"},
    "Governador Valadares": {"lat": -18.8500, "lon": -41.9500, "region": "Rio Doce"},
    "Ipatinga": {"lat": -19.4833, "lon": -42.5333, "region": "Rio Doce"},
    "Teófilo Otoni": {"lat": -17.8667, "lon": -41.5000, "region": "Jequitinhonha"},
    "Mantena": {"lat": -18.7833, "lon": -40.9833, "region": "Rio Doce"},
    "Aimorés": {"lat": -19.5000, "lon": -41.0667, "region": "Rio Doce"},
    "Caratinga": {"lat": -19.8000, "lon": -42.1333, "region": "Rio Doce"},
    "Belo Horizonte": {"lat": -19.9167, "lon": -43.9333, "region": "Metropolitana"},
    "Contagem": {"lat": -19.9322, "lon": -44.0539, "region": "Metropolitana"},
    "Betim": {"lat": -19.9667, "lon": -44.2000, "region": "Metropolitana"},
    "Nova Lima": {"lat": -19.9833, "lon": -43.8500, "region": "Metropolitana"},
    "Sabará": {"lat": -19.8833, "lon": -43.8000, "region": "Metropolitana"},
    "Juiz de Fora": {"lat": -21.7500, "lon": -43.3500, "region": "Mata"},
    "Ubá": {"lat": -21.1167, "lon": -42.9333, "region": "Mata"},
    "Cataguases": {"lat": -21.3833, "lon": -42.6833, "region": "Mata"},
    "Leopoldina": {"lat": -21.5333, "lon": -42.6500, "region": "Mata"},
    "Muriaé": {"lat": -21.1333, "lon": -42.3667, "region": "Mata"},
    "Divinópolis": {"lat": -20.1333, "lon": -44.8833, "region": "Centro-Oeste"},
    "Itaúna": {"lat": -20.0667, "lon": -44.5667, "region": "Centro-Oeste"},
    "Pará de Minas": {"lat": -19.8500, "lon": -44.6000, "region": "Centro-Oeste"},
    "Diamantina": {"lat": -18.2500, "lon": -43.6000, "region": "Jequitinhonha"},
    "Serro": {"lat": -18.6000, "lon": -43.3833, "region": "Jequitinhonha"},
    "Paracatu": {"lat": -17.2167, "lon": -46.8667, "region": "Noroeste"},
    "Unaí": {"lat": -16.3500, "lon": -46.9000, "region": "Noroeste"},
    "Buritis": {"lat": -15.6167, "lon": -46.4167, "region": "Noroeste"},
}

# ==================== FUNÇÕES GEOGRÁFICAS ====================

def lat_lon_to_screen(lat, lon, zoom=1.0, offset_x=0, offset_y=0):
    y_ratio = (lat - MG_LAT_MIN) / (MG_LAT_MAX - MG_LAT_MIN)
    y = int(y_ratio * HEIGHT)
    y = HEIGHT - y
    x_ratio = (lon - MG_LON_MIN) / (MG_LON_MAX - MG_LON_MIN)
    x = int(x_ratio * MAP_WIDTH)
    center_x = MAP_WIDTH / 2
    center_y = HEIGHT / 2
    x = center_x + (x - center_x) * zoom + offset_x
    y = center_y + (y - center_y) * zoom + offset_y
    return (int(x), int(y))

def calculate_real_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ==================== CLASSES ====================

class Patient:
    def __init__(self, id, name, city_name, lat, lon, urgency_level, medical_category, 
                 estimated_service_time=30, address="", phone=""):
        self.id = id
        self.name = name
        self.city_name = city_name
        self.lat = lat
        self.lon = lon
        self.urgency_level = urgency_level
        self.medical_category = medical_category
        self.estimated_service_time = estimated_service_time
        self.address = address or f"{city_name}, MG"
        self.phone = phone
        self.priority_score = self.calculate_priority_score()
        self.screen_pos = None
        
    def get_color(self):
        colors = {
            UrgencyLevel.EMERGENCY: COLOR_EMERGENCY,
            UrgencyLevel.HIGH: COLOR_HIGH,
            UrgencyLevel.MEDIUM: COLOR_MEDIUM,
            UrgencyLevel.LOW: COLOR_LOW,
            UrgencyLevel.ROUTINE: COLOR_ROUTINE
        }
        return colors.get(self.urgency_level, COLOR_ROUTINE)
    
    def calculate_priority_score(self):
        """Calcula pontuação de prioridade (0-100) com pesos diferenciados"""
        # Base: valor do nível de urgência (5,4,3,2,1) * 15 = 75,60,45,30,15
        base_score = self.urgency_level.value * 15
        
        # Bônus para categorias especiais (saúde da mulher)
        bonuses = {
            MedicalCategory.DOMESTIC_VIOLENCE: 30,  # Violência doméstica - prioridade máxima
            MedicalCategory.EMERGENCY: 28,          # Emergência médica
            MedicalCategory.POSTPARTUM: 25,         # Pós-parto
            MedicalCategory.PREGNANCY: 20,          # Gestação
            MedicalCategory.REMOTE_AREA: 15,        # Área remota
            MedicalCategory.PRENATAL: 12,           # Pré-natal
            MedicalCategory.HORMONAL: 8,            # Hormonal
            MedicalCategory.ROUTINE_CHECKUP: 5      # Rotina
        }
        
        base_score += bonuses.get(self.medical_category, 0)
        
        # Bônus para cidades remotas (dificuldade de acesso)
        remote_cities = ["Manga", "Porteirinha", "Salinas", "Buritis", "Paracatu", "Unaí", "Diamantina", "Serro"]
        if self.city_name in remote_cities:
            base_score += 10
            
        # Garantir que não ultrapasse 100
        return min(100, base_score)
    
    def update_screen_pos(self, zoom, offset_x, offset_y):
        self.screen_pos = lat_lon_to_screen(self.lat, self.lon, zoom, offset_x, offset_y)
    
    def __eq__(self, other):
        if not isinstance(other, Patient):
            return False
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)

# ==================== TELA DE CONFIGURAÇÃO ====================

class ConfigScreen:
    def __init__(self, screen):
        self.screen = screen
        
        # Fontes
        self.title_font = pygame.font.SysFont('segoeui', 34, bold=True)
        self.subtitle_font = pygame.font.SysFont('segoeui', 20, bold=True)
        self.label_font = pygame.font.SysFont('segoeui', 16, bold=True)
        self.normal_font = pygame.font.SysFont('segoeui', 14)
        self.value_font = pygame.font.SysFont('segoeui', 24, bold=True)
        self.small_font = pygame.font.SysFont('segoeui', 12)
        
        # Parâmetros
        self.population_size = 200
        self.num_generations = 300
        self.mutation_rate = 0.35
        self.num_patients = 20
        
        # Métodos selecionados
        self.init_method = InitializationMethod.PRIORITY  # Mudado para Prioridade Pura como padrão
        self.selection_method = SelectionMethod.ELITIST_TOURNAMENT  # Mudado para Elitista
        
        # Listas de métodos
        self.init_methods = list(InitializationMethod)
        self.selection_methods = list(SelectionMethod)
        self.current_init_index = 1  # PRIORITY
        self.current_sel_index = 1   # ELITIST_TOURNAMENT
        
        # Layout - Centralizado
        center_x = WIDTH // 2
        self.card_width = 400
        self.card_x = center_x - self.card_width // 2
        
        # Posições Y
        self.title_y = 40
        self.subtitle_y = 85
        
        # População
        self.population_y = 140
        # Gerações
        self.generations_y = 230
        # Pacientes
        self.patients_y = 320
        # Inicialização
        self.init_y = 420
        # Seleção
        self.selection_y = 500
        # Mutação
        self.mutation_y = 580
        
        # Botões
        button_width = 45
        button_height = 45
        self.population_decr = pygame.Rect(self.card_x + 20, self.population_y + 20, button_width, button_height)
        self.population_incr = pygame.Rect(self.card_x + self.card_width - 65, self.population_y + 20, button_width, button_height)
        self.generations_decr = pygame.Rect(self.card_x + 20, self.generations_y + 20, button_width, button_height)
        self.generations_incr = pygame.Rect(self.card_x + self.card_width - 65, self.generations_y + 20, button_width, button_height)
        self.patients_decr = pygame.Rect(self.card_x + 20, self.patients_y + 20, button_width, button_height)
        self.patients_incr = pygame.Rect(self.card_x + self.card_width - 65, self.patients_y + 20, button_width, button_height)
        
        # Botões de métodos
        self.init_prev = pygame.Rect(self.card_x + 20, self.init_y + 20, button_width, button_height)
        self.init_next = pygame.Rect(self.card_x + self.card_width - 65, self.init_y + 20, button_width, button_height)
        self.sel_prev = pygame.Rect(self.card_x + 20, self.selection_y + 20, button_width, button_height)
        self.sel_next = pygame.Rect(self.card_x + self.card_width - 65, self.selection_y + 20, button_width, button_height)
        
        # Slider de mutação
        self.mutation_slider_rect = pygame.Rect(self.card_x + 20, self.mutation_y + 30, self.card_width - 40, 8)
        self.mutation_knob_radius = 12
        self.dragging_slider = False
        
        # Botão iniciar
        self.start_button = pygame.Rect(center_x - 120, HEIGHT - 90, 240, 55)
        
        # Painel de informações
        self.info_panel_rect = pygame.Rect(center_x - 450, HEIGHT - 210, 900, 160)
        
    def draw_card(self, y, label, value, description=""):
        card_rect = pygame.Rect(self.card_x, y, self.card_width, 70)
        pygame.draw.rect(self.screen, CARD_BG, card_rect, border_radius=12)
        pygame.draw.rect(self.screen, PRIMARY_COLOR, card_rect, 2, border_radius=12)
        
        label_surf = self.label_font.render(label, True, PRIMARY_COLOR)
        self.screen.blit(label_surf, (self.card_x + 20, y + 12))
        
        if description:
            desc_surf = self.small_font.render(description, True, MEDIUM_GRAY)
            self.screen.blit(desc_surf, (self.card_x + 20, y + 32))
        
        value_surf = self.value_font.render(str(value), True, SUCCESS_COLOR)
        value_rect = value_surf.get_rect(center=(self.card_x + self.card_width // 2, y + 40))
        self.screen.blit(value_surf, value_rect)
        
        return card_rect
    
    def draw_method_card(self, y, label, current_method, description=""):
        card_rect = pygame.Rect(self.card_x, y, self.card_width, 70)
        pygame.draw.rect(self.screen, CARD_BG, card_rect, border_radius=12)
        pygame.draw.rect(self.screen, SUCCESS_COLOR, card_rect, 2, border_radius=12)
        
        label_surf = self.label_font.render(label, True, PRIMARY_COLOR)
        self.screen.blit(label_surf, (self.card_x + 20, y + 12))
        
        if description:
            desc_surf = self.small_font.render(description, True, MEDIUM_GRAY)
            self.screen.blit(desc_surf, (self.card_x + 20, y + 32))
        
        method_surf = self.normal_font.render(current_method.value, True, WHITE)
        method_rect = method_surf.get_rect(center=(self.card_x + self.card_width // 2, y + 40))
        self.screen.blit(method_surf, method_rect)
        
        return card_rect
    
    def draw_button(self, rect, text, color, hover_color=None, text_color=WHITE):
        mouse_pos = pygame.mouse.get_pos()
        is_hover = rect.collidepoint(mouse_pos)
        current_color = hover_color if is_hover and hover_color else color
        
        pygame.draw.rect(self.screen, current_color, rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, rect, 2, border_radius=10)
        
        if rect.width < 60:
            font = pygame.font.SysFont('segoeui', 24, bold=True)
        else:
            font = self.label_font
        
        text_surf = font.render(text, True, text_color)
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)
        
        return is_hover
    
    def draw_slider(self):
        label_surf = self.label_font.render("Taxa de Mutacao", True, PRIMARY_COLOR)
        self.screen.blit(label_surf, (self.card_x + 20, self.mutation_y + 12))
        
        desc_surf = self.small_font.render("(maior = mais diversidade, menor = mais convergencia)", True, MEDIUM_GRAY)
        self.screen.blit(desc_surf, (self.card_x + 20, self.mutation_y + 32))
        
        pygame.draw.rect(self.screen, DARK_GRAY, self.mutation_slider_rect, border_radius=5)
        
        progress = (self.mutation_rate - 0.1) / 0.7
        filled_width = self.mutation_slider_rect.width * progress
        filled_rect = pygame.Rect(self.mutation_slider_rect.x, self.mutation_slider_rect.y, 
                                  filled_width, self.mutation_slider_rect.height)
        pygame.draw.rect(self.screen, PRIMARY_COLOR, filled_rect, border_radius=5)
        
        knob_x = self.mutation_slider_rect.x + filled_width
        pygame.draw.circle(self.screen, WHITE, (int(knob_x), self.mutation_slider_rect.centery), 
                          self.mutation_knob_radius)
        pygame.draw.circle(self.screen, PRIMARY_COLOR, (int(knob_x), self.mutation_slider_rect.centery), 
                          self.mutation_knob_radius - 3)
        
        value_surf = self.value_font.render(f"{self.mutation_rate:.2f}", True, SUCCESS_COLOR)
        value_rect = value_surf.get_rect(center=(self.card_x + self.card_width - 40, self.mutation_y + 45))
        self.screen.blit(value_surf, value_rect)
    
    def draw_info_panel(self):
        pygame.draw.rect(self.screen, (0, 0, 0, 200), self.info_panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, PRIMARY_COLOR, self.info_panel_rect, 2, border_radius=12)
        
        title_surf = self.small_font.render("INFORMACOES DO SISTEMA", True, PRIMARY_COLOR)
        self.screen.blit(title_surf, (self.info_panel_rect.x + 15, self.info_panel_rect.y + 10))
        
        left_x = self.info_panel_rect.x + 15
        right_x = self.info_panel_rect.x + self.info_panel_rect.width // 2 + 10
        y_offset = self.info_panel_rect.y + 35
        line_height = 22
        
        info_lines_left = [
            f"• Cidades disponiveis: {len(CITIES_MG)} municipios",
            "• CADA CIDADE APARECE APENAS UMA VEZ",
            "• ALTA PRIORIZACAO para atendimentos URGENTES",
            "• Emergencias e Alta prioridade sao atendidas PRIMEIRO",
            "• Metodos de Inicializacao:",
            "  - Prioridade Pura (RECOMENDADO)"
        ]
        
        info_lines_right = [
            "• Metodos de Selecao:",
            "  - Torneio Elitista (RECOMENDADO)",
            "  - Torneio, Roleta, Ranking, Boltzmann",
            "• Zoom: Scroll do mouse (0.5x a 5.0x)",
            "• Pan: Clique e arraste no mapa",
            "• Fitness: MENOR valor = MELHOR rota"
        ]
        
        for i, line in enumerate(info_lines_left):
            color = LIGHT_GRAY if "•" in line else MEDIUM_GRAY
            text = self.small_font.render(line, True, color)
            self.screen.blit(text, (left_x, y_offset + i * line_height))
        
        for i, line in enumerate(info_lines_right):
            color = LIGHT_GRAY if "•" in line else MEDIUM_GRAY
            text = self.small_font.render(line, True, color)
            self.screen.blit(text, (right_x, y_offset + i * line_height))
    
    def run(self):
        clock = pygame.time.Clock()
        
        while True:
            self.screen.fill(BLACK)
            
            title_surf = self.title_font.render("SISTEMA DE ROTEAMENTO MEDICO", True, PRIMARY_COLOR)
            title_rect = title_surf.get_rect(center=(WIDTH//2, self.title_y))
            self.screen.blit(title_surf, title_rect)
            
            subtitle_surf = self.subtitle_font.render("Minas Gerais - Saude da Mulher", True, LIGHT_GRAY)
            subtitle_rect = subtitle_surf.get_rect(center=(WIDTH//2, self.subtitle_y))
            self.screen.blit(subtitle_surf, subtitle_rect)
            
            pygame.draw.line(self.screen, PRIMARY_COLOR, (WIDTH//4, self.subtitle_y + 25), (3*WIDTH//4, self.subtitle_y + 25), 2)
            
            self.draw_card(self.population_y, "Tamanho da Populacao", self.population_size, "mais diversidade, porem mais lento")
            self.draw_card(self.generations_y, "Numero de Geracoes", self.num_generations, "mais iteracoes = melhor resultado")
            self.draw_card(self.patients_y, "Pacientes a Atender", self.num_patients, "cidades a serem visitadas (sem repeticoes)")
            
            self.draw_method_card(self.init_y, "Metodo de Inicializacao", self.init_method, "")
            self.draw_method_card(self.selection_y, "Metodo de Selecao", self.selection_method, "")
            
            self.draw_slider()
            self.draw_info_panel()
            
            self.draw_button(self.population_decr, "-", SECONDARY_COLOR, (180, 30, 30))
            self.draw_button(self.population_incr, "+", SUCCESS_COLOR, (30, 180, 80))
            self.draw_button(self.generations_decr, "-", SECONDARY_COLOR, (180, 30, 30))
            self.draw_button(self.generations_incr, "+", SUCCESS_COLOR, (30, 180, 80))
            self.draw_button(self.patients_decr, "-", SECONDARY_COLOR, (180, 30, 30))
            self.draw_button(self.patients_incr, "+", SUCCESS_COLOR, (30, 180, 80))
            
            self.draw_button(self.init_prev, "<", SECONDARY_COLOR, (180, 30, 30))
            self.draw_button(self.init_next, ">", SUCCESS_COLOR, (30, 180, 80))
            self.draw_button(self.sel_prev, "<", SECONDARY_COLOR, (180, 30, 30))
            self.draw_button(self.sel_next, ">", SUCCESS_COLOR, (30, 180, 80))
            
            self.draw_button(self.start_button, "INICIAR OTIMIZACAO", SUCCESS_COLOR, (70, 220, 120))
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.population_decr.collidepoint(event.pos):
                            self.population_size = max(50, self.population_size - 10)
                        elif self.population_incr.collidepoint(event.pos):
                            self.population_size = min(500, self.population_size + 10)
                        elif self.generations_decr.collidepoint(event.pos):
                            self.num_generations = max(50, self.num_generations - 10)
                        elif self.generations_incr.collidepoint(event.pos):
                            self.num_generations = min(500, self.num_generations + 10)
                        elif self.patients_decr.collidepoint(event.pos):
                            self.num_patients = max(10, self.num_patients - 2)
                        elif self.patients_incr.collidepoint(event.pos):
                            self.num_patients = min(40, self.num_patients + 2)
                        elif self.init_prev.collidepoint(event.pos):
                            self.current_init_index = (self.current_init_index - 1) % len(self.init_methods)
                            self.init_method = self.init_methods[self.current_init_index]
                        elif self.init_next.collidepoint(event.pos):
                            self.current_init_index = (self.current_init_index + 1) % len(self.init_methods)
                            self.init_method = self.init_methods[self.current_init_index]
                        elif self.sel_prev.collidepoint(event.pos):
                            self.current_sel_index = (self.current_sel_index - 1) % len(self.selection_methods)
                            self.selection_method = self.selection_methods[self.current_sel_index]
                        elif self.sel_next.collidepoint(event.pos):
                            self.current_sel_index = (self.current_sel_index + 1) % len(self.selection_methods)
                            self.selection_method = self.selection_methods[self.current_sel_index]
                        elif self.mutation_slider_rect.collidepoint(event.pos):
                            self.dragging_slider = True
                        elif self.start_button.collidepoint(event.pos):
                            return {
                                'population_size': self.population_size,
                                'num_generations': self.num_generations,
                                'mutation_rate': self.mutation_rate,
                                'num_patients': self.num_patients,
                                'init_method': self.init_method,
                                'selection_method': self.selection_method
                            }
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging_slider = False
                elif event.type == pygame.MOUSEMOTION and self.dragging_slider:
                    relative_x = max(0, min(self.mutation_slider_rect.width, 
                                           event.pos[0] - self.mutation_slider_rect.x))
                    self.mutation_rate = 0.1 + (relative_x / self.mutation_slider_rect.width) * 0.7
                    self.mutation_rate = round(max(0.1, min(0.8, self.mutation_rate)), 2)
            
            pygame.display.flip()
            clock.tick(FPS)

# ==================== ALGORITMO GENÉTICO COM ALTA PRIORIZAÇÃO ====================

class PriorityGeneticAlgorithm:
    def __init__(self, patients, population_size=200, mutation_rate=0.35, 
                 crossover_rate=0.8, elitism_count=5,
                 init_method=InitializationMethod.PRIORITY,
                 selection_method=SelectionMethod.ELITIST_TOURNAMENT):
        self.patients = patients
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.init_method = init_method
        self.selection_method = selection_method
        
        self.population = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_distance = float('inf')
        self.fitness_history = []
        self.distance_history = []
        self.generation = 0
        self.priority_metrics = {'emergency_first': 0, 'avg_priority_position': 0, 'priority_coverage': 0}
        
    def calculate_distance_km(self, p1, p2):
        return calculate_real_distance(p1.lat, p1.lon, p2.lat, p2.lon)
    
    def calculate_total_distance_km(self, route):
        if not route or len(route) < 2:
            return 0
        total = 0
        for i in range(len(route)):
            total += self.calculate_distance_km(route[i], route[(i + 1) % len(route)])
        return total
    
    def calculate_fitness(self, route):
        """
        Calcula o fitness da rota.
        QUANTO MENOR O FITNESS, MELHOR A SOLUÇÃO!
        
        O fitness é composto por:
        1. Distância total (peso baixo)
        2. Benefício da prioridade (subtrai do fitness - quanto maior a prioridade, menor o fitness)
        3. Penalidade de posição (força prioridades para o início)
        4. Penalidade de ordem (penaliza se emergências estão no final)
        """
        if not route:
            return float('inf')
        
        n = len(route)
        total_distance = 0
        total_priority = 0
        position_penalty = 0
        urgency_order_penalty = 0
        
        # Variáveis para rastrear a posição das prioridades
        emergency_positions = []
        high_priority_positions = []
        
        for i, patient in enumerate(route):
            # Soma da prioridade (quanto maior, melhor - reduz o fitness)
            total_priority += patient.priority_score
            
            # Penalidade para emergências (peso maior)
            if patient.urgency_level == UrgencyLevel.EMERGENCY:
                emergency_positions.append(i)
                # Penalidade EXTREMA se emergência está no final
                position_factor = (n - i) / n
                urgency_order_penalty += patient.priority_score * position_factor * FITNESS_WEIGHTS['urgency_penalty']
            
            # Penalidade para alta prioridade
            elif patient.urgency_level == UrgencyLevel.HIGH:
                high_priority_positions.append(i)
                position_factor = (n - i) / n
                urgency_order_penalty += patient.priority_score * position_factor * FITNESS_WEIGHTS['position_penalty']
            
            # Acumular distância
            if i > 0:
                dist = self.calculate_distance_km(route[i-1], route[i])
                total_distance += dist
        
        # Penalidade adicional se as primeiras posições não são de alta prioridade
        if n > 0:
            first_patient = route[0]
            if first_patient.urgency_level not in [UrgencyLevel.EMERGENCY, UrgencyLevel.HIGH]:
                position_penalty += 50  # Penalidade forte se o primeiro não é prioritário
        
        # Penalidade se não há emergências nos primeiros 20% da rota
        emergency_in_first_20 = any(i < n * 0.2 for i in emergency_positions)
        if not emergency_in_first_20 and emergency_positions:
            position_penalty += 100  # Penalidade EXTREMA
        
        # Cálculo final do fitness (MENOR é MELHOR)
        distance_cost = (total_distance / 100) * FITNESS_WEIGHTS['distance_weight']
        priority_benefit = (total_priority / 10) * FITNESS_WEIGHTS['priority_weight']
        total_penalties = urgency_order_penalty + position_penalty
        
        fitness = distance_cost - priority_benefit + total_penalties
        
        return max(0.01, fitness)
    
    def calculate_priority_metrics(self, route):
        """Calcula métricas de qualidade da priorização"""
        n = len(route)
        if n == 0:
            return
        
        # Posição média de emergências
        emergency_positions = [i for i, p in enumerate(route) if p.urgency_level == UrgencyLevel.EMERGENCY]
        high_priority_positions = [i for i, p in enumerate(route) if p.urgency_level == UrgencyLevel.HIGH]
        
        if emergency_positions:
            self.priority_metrics['avg_emergency_position'] = sum(emergency_positions) / len(emergency_positions) / n
            self.priority_metrics['emergency_in_first_20'] = any(i < n * 0.2 for i in emergency_positions)
        else:
            self.priority_metrics['avg_emergency_position'] = 1
            self.priority_metrics['emergency_in_first_20'] = False
        
        if high_priority_positions:
            self.priority_metrics['avg_high_position'] = sum(high_priority_positions) / len(high_priority_positions) / n
        else:
            self.priority_metrics['avg_high_position'] = 1
        
        # Porcentagem de prioridades nos primeiros 30%
        total_priorities = len(emergency_positions) + len(high_priority_positions)
        priorities_in_first_30 = sum(1 for i in emergency_positions + high_priority_positions if i < n * 0.3)
        
        self.priority_metrics['priority_coverage'] = priorities_in_first_30 / total_priorities if total_priorities > 0 else 1
        self.priority_metrics['total_emergencies'] = len(emergency_positions)
        self.priority_metrics['total_high_priority'] = len(high_priority_positions)
    
    def initialize_population_priority_pure(self, unique_cities):
        """Inicialização PRIORIDADE PURA - TODAS as rotas começam ordenadas por prioridade"""
        # Ordenar estritamente por prioridade (decrescente)
        route = sorted(unique_cities, key=lambda p: (-p.priority_score, -p.urgency_level.value))
        
        # Pequena perturbação para criar diversidade, mas mantendo a estrutura de prioridade
        # Permite trocar apenas pacientes de prioridade similar
        swap_count = random.randint(1, max(1, len(route) // 20))
        for _ in range(swap_count):
            # Encontrar dois pacientes com prioridade similar (±15 pontos)
            for _ in range(50):  # Tentativas limitadas
                i, j = random.sample(range(len(route)), 2)
                if abs(route[i].priority_score - route[j].priority_score) <= 15:
                    route[i], route[j] = route[j], route[i]
                    break
        
        return route
    
    def initialize_population_random(self, unique_cities):
        return random.sample(unique_cities, len(unique_cities))
    
    def initialize_population_nearest_neighbor(self, unique_cities):
        remaining = unique_cities.copy()
        # Começar pelo paciente de maior prioridade
        start = max(remaining, key=lambda p: p.priority_score)
        route = [start]
        remaining.remove(start)
        
        while remaining:
            current = route[-1]
            # Escolher próximo baseado em distância E prioridade
            next_patient = min(remaining, key=lambda p: 
                self.calculate_distance_km(current, p) / (p.priority_score / 100 + 0.1))
            route.append(next_patient)
            remaining.remove(next_patient)
        
        return route
    
    def initialize_population_hybrid(self, unique_cities):
        # 80% prioridade pura, 20% nearest neighbor
        if random.random() < 0.8:
            return self.initialize_population_priority_pure(unique_cities)
        else:
            return self.initialize_population_nearest_neighbor(unique_cities)
    
    def initialize_population_cluster(self, unique_cities):
        # Agrupar por região, depois ordenar cada região por prioridade
        regions = {}
        for p in unique_cities:
            region = CITIES_MG[p.city_name]["region"]
            if region not in regions:
                regions[region] = []
            regions[region].append(p)
        
        # Ordenar cada região por prioridade
        for region in regions:
            regions[region].sort(key=lambda p: -p.priority_score)
        
        # Intercalar regiões, priorizando regiões com maior prioridade média
        route = []
        max_region_size = max(len(r) for r in regions.values())
        for i in range(max_region_size):
            for region in sorted(regions.keys(), key=lambda r: -sum(p.priority_score for p in regions[r]) / len(regions[r])):
                if i < len(regions[region]):
                    route.append(regions[region][i])
        
        return route
    
    def initialize_population(self):
        """Inicializa população com foco em prioridade"""
        self.population = []
        unique_cities = list({p.city_name: p for p in self.patients}.values())
        
        print(f"\nInicializando população com método: {self.init_method.value}")
        print("Ênfase em PRIORIDADE - atendimentos urgentes serão priorizados!\n")
        
        for i in range(self.population_size):
            if self.init_method == InitializationMethod.RANDOM:
                route = self.initialize_population_random(unique_cities)
            elif self.init_method == InitializationMethod.PRIORITY:
                route = self.initialize_population_priority_pure(unique_cities)
            elif self.init_method == InitializationMethod.NEAREST_NEIGHBOR:
                route = self.initialize_population_nearest_neighbor(unique_cities)
            elif self.init_method == InitializationMethod.HYBRID:
                route = self.initialize_population_hybrid(unique_cities)
            elif self.init_method == InitializationMethod.CLUSTER:
                route = self.initialize_population_cluster(unique_cities)
            else:
                route = self.initialize_population_priority_pure(unique_cities)
            
            self.population.append(route)
    
    def select_parent_tournament(self, fitness_values):
        tournament_size = 5
        indices = random.sample(range(len(self.population)), tournament_size)
        best_idx = min(indices, key=lambda i: fitness_values[i])
        return copy.deepcopy(self.population[best_idx])
    
    def select_parent_elitist_tournament(self, fitness_values):
        # 70% de chance de pegar do top 20% (elite)
        if random.random() < 0.7:
            elite_size = max(1, len(self.population) // 5)
            elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])[:elite_size]
            return copy.deepcopy(self.population[random.choice(elite_indices)])
        else:
            return self.select_parent_tournament(fitness_values)
    
    def select_parent_roulette(self, fitness_values):
        max_fitness = max(fitness_values)
        adjusted_fitness = [max_fitness - f + 0.1 for f in fitness_values]
        total_fitness = sum(adjusted_fitness)
        probabilities = [f / total_fitness for f in adjusted_fitness]
        idx = random.choices(range(len(self.population)), weights=probabilities, k=1)[0]
        return copy.deepcopy(self.population[idx])
    
    def select_parent_rank(self, fitness_values):
        sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])
        ranks = list(range(len(sorted_indices), 0, -1))
        total_rank = sum(ranks)
        probabilities = [r / total_rank for r in ranks]
        idx = random.choices(sorted_indices, weights=probabilities, k=1)[0]
        return copy.deepcopy(self.population[idx])
    
    def select_parent_boltzmann(self, fitness_values):
        temperature = max(0.1, 1.0 - (self.generation / 500))
        adjusted = [math.exp(-f / temperature) for f in fitness_values]
        total = sum(adjusted)
        if total > 0:
            probabilities = [a / total for a in adjusted]
            idx = random.choices(range(len(self.population)), weights=probabilities, k=1)[0]
        else:
            idx = random.randint(0, len(self.population) - 1)
        return copy.deepcopy(self.population[idx])
    
    def select_parent(self, fitness_values):
        if self.selection_method == SelectionMethod.TOURNAMENT:
            return self.select_parent_tournament(fitness_values)
        elif self.selection_method == SelectionMethod.ELITIST_TOURNAMENT:
            return self.select_parent_elitist_tournament(fitness_values)
        elif self.selection_method == SelectionMethod.ROULETTE:
            return self.select_parent_roulette(fitness_values)
        elif self.selection_method == SelectionMethod.RANK:
            return self.select_parent_rank(fitness_values)
        elif self.selection_method == SelectionMethod.BOLTZMANN:
            return self.select_parent_boltzmann(fitness_values)
        else:
            return self.select_parent_elitist_tournament(fitness_values)
    
    def crossover(self, parent1, parent2):
        """Crossover que preserva a ordem das prioridades"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)
        
        n = len(parent1)
        
        # Identificar pacientes de altíssima prioridade (emergência e prioridade > 80)
        critical_patients = set()
        for p in parent1:
            if p.urgency_level == UrgencyLevel.EMERGENCY or p.priority_score > 80:
                critical_patients.add(p.city_name)
        
        child = [None] * n
        
        # Preservar posições dos pacientes críticos do parent1
        for i, patient in enumerate(parent1):
            if patient.city_name in critical_patients:
                child[i] = patient
        
        # Preencher o resto com parent2
        remaining = []
        used_cities = set(p.city_name for p in child if p is not None)
        
        # Priorizar manter a ordem de prioridade do parent2
        for patient in parent2:
            if patient.city_name not in used_cities:
                remaining.append(patient)
        
        pos = 0
        for i in range(n):
            if child[i] is None:
                child[i] = remaining[pos]
                pos += 1
        
        return child
    
    def mutate(self, route):
        """Mutações que respeitam a prioridade - não misturam níveis diferentes de urgência"""
        if random.random() > self.mutation_rate:
            return copy.deepcopy(route)
        
        mutated = copy.deepcopy(route)
        n = len(mutated)
        
        if n < 2:
            return mutated
        
        # Identificar níveis de urgência
        emergency_indices = [i for i, p in enumerate(mutated) if p.urgency_level == UrgencyLevel.EMERGENCY]
        high_indices = [i for i, p in enumerate(mutated) if p.urgency_level == UrgencyLevel.HIGH]
        medium_indices = [i for i, p in enumerate(mutated) if p.urgency_level == UrgencyLevel.MEDIUM]
        low_indices = [i for i, p in enumerate(mutated) if p.urgency_level == UrgencyLevel.LOW]
        
        mutation_type = random.choice(['shift_emergency', 'swap_same_level', 'invert', 'local_sort'])
        
        if mutation_type == 'shift_emergency' and emergency_indices:
            # Mover emergência para mais perto do início
            idx = random.choice(emergency_indices)
            if idx > 0:
                patient = mutated.pop(idx)
                # Colocar nas primeiras posições
                new_pos = random.randint(0, min(3, idx))
                mutated.insert(new_pos, patient)
        
        elif mutation_type == 'swap_same_level' and n >= 2:
            # Trocar pacientes do mesmo nível de urgência
            level_indices = emergency_indices or high_indices or medium_indices or low_indices
            if len(level_indices) >= 2:
                i, j = random.sample(level_indices, 2)
                mutated[i], mutated[j] = mutated[j], mutated[i]
        
        elif mutation_type == 'invert' and n >= 3:
            # Inverter um pequeno segmento (mantém estrutura local)
            start = random.randint(0, n - 3)
            end = start + random.randint(2, min(4, n - start))
            mutated[start:end] = reversed(mutated[start:end])
        
        elif mutation_type == 'local_sort' and n >= 3:
            # Ordenar um pequeno segmento por prioridade
            start = random.randint(0, max(0, n - 4))
            end = min(n, start + random.randint(3, 5))
            segment = mutated[start:end]
            segment.sort(key=lambda p: -p.priority_score)
            mutated[start:end] = segment
        
        return mutated
    
    def evolve_generation(self):
        fitness_values = [self.calculate_fitness(route) for route in self.population]
        distance_values = [self.calculate_total_distance_km(route) for route in self.population]
        
        best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
        current_best = self.population[best_idx]
        current_fitness = fitness_values[best_idx]
        current_distance = distance_values[best_idx]
        
        improved = False
        if current_fitness < self.best_fitness:
            improved = True
            self.best_fitness = current_fitness
            self.best_distance = current_distance
            self.best_solution = copy.deepcopy(current_best)
            self.calculate_priority_metrics(self.best_solution)
        
        self.fitness_history.append(self.best_fitness)
        self.distance_history.append(self.best_distance)
        
        # Elitismo (manter os melhores)
        elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])[:self.elitism_count]
        new_population = [copy.deepcopy(self.population[i]) for i in elite_indices]
        
        while len(new_population) < self.population_size:
            parent1 = self.select_parent(fitness_values)
            parent2 = self.select_parent(fitness_values)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        return {'generation': self.generation, 'best_fitness': self.best_fitness, 
                'best_distance': self.best_distance, 'improved': improved,
                'priority_metrics': self.priority_metrics.copy()}

# ==================== VISUALIZADOR ====================

class RealTimeVisualizer:
    def __init__(self, screen):
        self.screen = screen
        self.animation_offset = 0
        self.pulse_alpha = 0
        self.pulse_direction = 1
        self.particles = []
        self.connection_fade = {}
        
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.drag_start = (0, 0)
        self.drag_start_offset = (0, 0)
        
        self.MIN_ZOOM = 0.5
        self.MAX_ZOOM = 5.0
        self.detail_level = "normal"
        
    def update_detail_level(self):
        if self.zoom < 0.8:
            self.detail_level = "low"
        elif self.zoom < 1.5:
            self.detail_level = "normal"
        elif self.zoom < 2.5:
            self.detail_level = "high"
        else:
            self.detail_level = "ultra"
    
    def handle_zoom_event(self, event):
        if event.type == pygame.MOUSEWHEEL:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if mouse_x < MAP_WIDTH:
                old_zoom = self.zoom
                self.zoom += event.y * 0.15
                self.zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self.zoom))
                zoom_ratio = self.zoom / old_zoom
                self.offset_x = mouse_x - (mouse_x - self.offset_x) * zoom_ratio
                self.offset_y = mouse_y - (mouse_y - self.offset_y) * zoom_ratio
                self.update_detail_level()
    
    def handle_pan_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if event.pos[0] < MAP_WIDTH:
                self.dragging = True
                self.drag_start = event.pos
                self.drag_start_offset = (self.offset_x, self.offset_y)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            dx = event.pos[0] - self.drag_start[0]
            dy = event.pos[1] - self.drag_start[1]
            self.offset_x = self.drag_start_offset[0] + dx
            self.offset_y = self.drag_start_offset[1] + dy
    
    def reset_view(self):
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.update_detail_level()
    
    def update_animation(self):
        self.animation_offset = (self.animation_offset + 0.05) % (2 * math.pi)
        self.pulse_alpha += 0.05 * self.pulse_direction
        if self.pulse_alpha >= 1 or self.pulse_alpha <= 0:
            self.pulse_direction *= -1
            self.pulse_alpha = max(0, min(1, self.pulse_alpha))
        
        for particle in self.particles[:]:
            particle['life'] -= 1
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            if particle['life'] <= 0:
                self.particles.remove(particle)
        
        for conn in list(self.connection_fade.keys()):
            self.connection_fade[conn] -= 5
            if self.connection_fade[conn] <= 0:
                del self.connection_fade[conn]
    
    def add_connection_highlight(self, p1, p2):
        key = (p1.id, p2.id)
        self.connection_fade[key] = 255
    
    def add_particle(self, x, y, color):
        self.particles.append({'x': x, 'y': y, 'vx': random.uniform(-2, 2), 
                              'vy': random.uniform(-2, 2), 'color': color, 'life': 30})
    
    def draw_route(self, route):
        if not route or len(route) < 2:
            return
        base_width = max(1, int(2 / max(0.5, self.zoom)))
        for i in range(len(route) - 1):
            p1 = route[i]
            p2 = route[i + 1]
            pos1 = p1.screen_pos
            pos2 = p2.screen_pos
            if pos1 and pos2:
                key = (p1.id, p2.id)
                if key in self.connection_fade:
                    color, width = (255, 255, 100), base_width + 2
                else:
                    color = p2.get_color()
                    width = base_width + int(p2.priority_score / 50)
                pygame.draw.line(self.screen, color, pos1, pos2, max(1, width))
                dist_km = calculate_real_distance(p1.lat, p1.lon, p2.lat, p2.lon)
                if self.detail_level in ["high", "ultra"] and dist_km > 50:
                    mid_x = (pos1[0] + pos2[0]) // 2
                    mid_y = (pos1[1] + pos2[1]) // 2
                    font_size = max(8, int(12 / max(0.5, self.zoom)))
                    font = pygame.font.SysFont('Arial', font_size, bold=True)
                    dist_text = font.render(f"{dist_km:.0f}km", True, WARNING_COLOR)
                    text_rect = dist_text.get_rect(center=(mid_x, mid_y - 10))
                    pygame.draw.rect(self.screen, (0, 0, 0, 150), text_rect.inflate(8, 4), border_radius=4)
                    self.screen.blit(dist_text, text_rect)
                if p2.urgency_level == UrgencyLevel.EMERGENCY:
                    pulse_intensity = int(100 + 155 * self.pulse_alpha)
                    pygame.draw.line(self.screen, (255, pulse_intensity, pulse_intensity), pos1, pos2, width + 2)
        if len(route) > 2:
            p1 = route[-1]
            p2 = route[0]
            if p1.screen_pos and p2.screen_pos:
                pygame.draw.line(self.screen, (80, 80, 100), p1.screen_pos, p2.screen_pos, 1)
    
    def draw_patient(self, patient, is_selected=False):
        pos = patient.screen_pos
        if not pos:
            return
        base_radius = NODE_RADIUS / max(0.5, self.zoom)
        if patient.urgency_level == UrgencyLevel.EMERGENCY:
            radius = max(6, int(base_radius + 4 / self.zoom))
        elif patient.urgency_level == UrgencyLevel.HIGH:
            radius = max(5, int(base_radius + 2 / self.zoom))
        else:
            radius = max(4, int(base_radius))
        if patient.urgency_level == UrgencyLevel.EMERGENCY:
            pulse_radius = radius + int(4 * self.pulse_alpha / max(0.5, self.zoom))
            pygame.draw.circle(self.screen, COLOR_EMERGENCY, pos, pulse_radius, 2)
        color = patient.get_color()
        pygame.draw.circle(self.screen, color, pos, radius)
        pygame.draw.circle(self.screen, WHITE, pos, radius, max(1, int(2 / self.zoom)))
        if is_selected:
            pygame.draw.circle(self.screen, (255, 255, 100), pos, radius + 3, max(1, int(3 / self.zoom)))
        icon_size = max(10, int(16 / max(0.5, self.zoom)))
        try:
            icon_font = pygame.font.SysFont('segoeui', icon_size)
            icon = icon_font.render(patient.urgency_level.icon, True, WHITE)
            icon_rect = icon.get_rect(center=pos)
            self.screen.blit(icon, icon_rect)
        except:
            pass
        if self.detail_level in ["normal", "high", "ultra"]:
            city_font_size = max(8, int(11 / max(0.5, self.zoom)))
            city_font = pygame.font.SysFont('Arial', city_font_size, bold=True)
            if patient.urgency_level == UrgencyLevel.EMERGENCY:
                text_color = COLOR_EMERGENCY
            elif patient.urgency_level == UrgencyLevel.HIGH:
                text_color = COLOR_HIGH
            else:
                text_color = WHITE
            city_text = city_font.render(patient.city_name, True, text_color)
            text_rect = city_text.get_rect(center=(pos[0], pos[1] - radius - 5))
            if self.zoom > 1.2:
                bg_rect = text_rect.inflate(6, 4)
                pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect, border_radius=4)
            if text_rect.x > 0 and text_rect.x < MAP_WIDTH:
                self.screen.blit(city_text, text_rect)
        if self.detail_level in ["high", "ultra"]:
            info_y_offset = radius + 5
            info_font_size = max(7, int(10 / max(0.5, self.zoom)))
            info_font = pygame.font.SysFont('Arial', info_font_size)
            patient_name = patient.name[:15] + "..." if len(patient.name) > 15 else patient.name
            name_text = info_font.render(patient_name, True, LIGHT_GRAY)
            name_rect = name_text.get_rect(center=(pos[0], pos[1] + info_y_offset))
            if name_rect.x > 0 and name_rect.x < MAP_WIDTH:
                self.screen.blit(name_text, name_rect)
            priority_text = info_font.render(f"Prio: {patient.priority_score:.0f}", True, SUCCESS_COLOR)
            priority_rect = priority_text.get_rect(center=(pos[0], pos[1] + info_y_offset + 12))
            if priority_rect.x > 0 and priority_rect.x < MAP_WIDTH:
                self.screen.blit(priority_text, priority_rect)
        if self.detail_level == "ultra":
            ultra_y_offset = radius + 28
            ultra_font_size = max(6, int(9 / max(0.5, self.zoom)))
            ultra_font = pygame.font.SysFont('Arial', ultra_font_size)
            cat_text = ultra_font.render(patient.medical_category.display_name[:12], True, MEDIUM_GRAY)
            cat_rect = cat_text.get_rect(center=(pos[0], pos[1] + ultra_y_offset))
            if cat_rect.x > 0 and cat_rect.x < MAP_WIDTH:
                self.screen.blit(cat_text, cat_rect)
            time_text = ultra_font.render(f"{patient.estimated_service_time}min", True, LIGHT_GRAY)
            time_rect = time_text.get_rect(center=(pos[0], pos[1] + ultra_y_offset + 10))
            if time_rect.x > 0 and time_rect.x < MAP_WIDTH:
                self.screen.blit(time_text, time_rect)
        if patient.urgency_level == UrgencyLevel.EMERGENCY and random.random() < 0.15:
            self.add_particle(pos[0] + random.randint(-radius, radius), 
                            pos[1] + random.randint(-radius, radius), COLOR_EMERGENCY)
        for particle in self.particles:
            particle_size = max(1, int(3 / max(0.5, self.zoom)))
            pygame.draw.circle(self.screen, particle['color'], 
                             (int(particle['x']), int(particle['y'])), particle_size)

# ==================== SISTEMA PRINCIPAL ====================

class HealthcareRoutingSystem:
    def __init__(self, population_size=200, num_generations=300, mutation_rate=0.35, 
                 num_patients=20, init_method=InitializationMethod.PRIORITY,
                 selection_method=SelectionMethod.ELITIST_TOURNAMENT):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Sistema de Roteamento Medico - MG (Priorizacao de Urgencias)")
        self.clock = pygame.time.Clock()
        
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.num_patients = num_patients
        self.init_method = init_method
        self.selection_method = selection_method
        
        # Fontes
        self.title_font = pygame.font.SysFont('segoeui', 22, bold=True)
        self.subtitle_font = pygame.font.SysFont('segoeui', 16, bold=True)
        self.normal_font = pygame.font.SysFont('segoeui', 13)
        self.small_font = pygame.font.SysFont('segoeui', 11)
        self.big_font = pygame.font.SysFont('segoeui', 28, bold=True)
        
        self.running = True
        self.paused = False
        self.selected_patient = None
        self.show_distance = True
        self.show_legend = True
        
        self.visualizer = RealTimeVisualizer(self.screen)
        self.patients = self.generate_unique_mg_patients(num_patients)
        self.initialize_algorithm()
        self.setup_ui()
        self.update_patient_positions()
        self.print_configuration()
    
    def initialize_algorithm(self):
        self.ga = PriorityGeneticAlgorithm(
            self.patients, 
            population_size=self.population_size,
            mutation_rate=self.mutation_rate,
            init_method=self.init_method,
            selection_method=self.selection_method
        )
        self.ga.initialize_population()
        self.paused = False
    
    def print_configuration(self):
        print("\n" + "="*70)
        print(" CONFIGURACOES DO ALGORITMO - PRIORIZACAO DE URGENCIAS")
        print("="*70)
        print(f" Tamanho da populacao: {self.population_size}")
        print(f" Numero de geracoes: {self.num_generations}")
        print(f" Taxa de mutacao: {self.mutation_rate:.2f}")
        print(f" Total de pacientes: {len(self.patients)}")
        print(f" Metodo de inicializacao: {self.init_method.value}")
        print(f" Metodo de selecao: {self.selection_method.value}")
        print(f" Cidades unicas: {len(set(p.city_name for p in self.patients))}")
        print("\n PESOS DO FITNESS (MENOR = MELHOR):")
        print(f"   - Distancia: peso {FITNESS_WEIGHTS['distance_weight']:.1f} (baixa importancia)")
        print(f"   - Prioridade: peso {FITNESS_WEIGHTS['priority_weight']:.1f} (ALTA importancia)")
        print(f"   - Penalidade de urgencia: {FITNESS_WEIGHTS['urgency_penalty']:.1f}")
        print(f"   - Penalidade de posicao: {FITNESS_WEIGHTS['position_penalty']:.1f}")
        print("\n PRIORIZACAO DE ATENDIMENTOS:")
        print("   🔴 EMERGENCIA → Atendidas PRIMEIRO (penalidade maxima se no final)")
        print("   🟠 ALTA PRIORIDADE → Atendidas no inicio")
        print("   🟡 MEDIA PRIORIDADE → Prioridade moderada")
        print("   🟢 BAIXA PRIORIDADE → Atendidas depois")
        print("   ⚪ ROTINA → Atendidas por ultimo")
        print("\n CONTROLES:")
        print("   Scroll - Zoom in/out")
        print("   Clique e arraste - Pan (mover o mapa)")
        print("   R - Reset algoritmo | CTRL+R - Reset view")
        print("   D - Mostrar distancia | L - Mostrar legenda")
        print("   C - Voltar para configuracao")
        print("="*70 + "\n")
    
    def update_patient_positions(self):
        for patient in self.patients:
            patient.update_screen_pos(self.visualizer.zoom, self.visualizer.offset_x, self.visualizer.offset_y)
        if self.ga.best_solution:
            for patient in self.ga.best_solution:
                patient.update_screen_pos(self.visualizer.zoom, self.visualizer.offset_x, self.visualizer.offset_y)
    
    def generate_unique_mg_patients(self, n):
        names = ["Ana Silva", "Maria Santos", "Carla Oliveira", "Patricia Souza", "Fernanda Lima",
                "Juliana Costa", "Renata Ferreira", "Beatriz Almeida", "Luciana Rodrigues", "Marcia Pereira",
                "Tatiana Gomes", "Simone Carvalho", "Andreia Ribeiro", "Camila Dias", "Vanessa Nunes",
                "Elaine Martins", "Cristina Rocha", "Larissa Mendes", "Sabrina Castro", "Michele Alves"]
        categories = list(MedicalCategory)
        patients = []
        max_cities = len(CITIES_MG)
        if n > max_cities:
            n = max_cities
        all_cities = list(CITIES_MG.keys())
        selected_cities = random.sample(all_cities, n)
        for i, city_name in enumerate(selected_cities):
            city_data = CITIES_MG[city_name]
            weights = [0.20, 0.12, 0.15, 0.05, 0.18, 0.08, 0.12, 0.10]
            category = random.choices(categories, weights=weights)[0]
            if category == MedicalCategory.EMERGENCY:
                urgency = UrgencyLevel.EMERGENCY
            elif category == MedicalCategory.DOMESTIC_VIOLENCE:
                urgency = UrgencyLevel.HIGH
            elif category == MedicalCategory.POSTPARTUM:
                urgency = random.choices([UrgencyLevel.HIGH, UrgencyLevel.MEDIUM], weights=[0.6, 0.4])[0]
            else:
                urgency = random.choices(list(UrgencyLevel), weights=[0.10, 0.15, 0.25, 0.25, 0.25])[0]
            patient = Patient(i, names[i % len(names)], city_name, city_data["lat"], city_data["lon"],
                            urgency, category, random.randint(20, 60), f"{city_name}, MG",
                            f"(31) 9{random.randint(1000,9999)}-{random.randint(1000,9999)}")
            patients.append(patient)
        return patients
    
    def setup_ui(self):
        button_y = 20
        self.buttons = []
        
        def toggle_pause():
            self.paused = not self.paused
        self.buttons.append({'rect': pygame.Rect(CONTROL_START_X + 10, button_y, 70, 35),
                            'text': '[Pause]' if not self.paused else '[Start]',
                            'color': PRIMARY_COLOR, 'action': toggle_pause})
        
        def reset_algorithm():
            self.initialize_algorithm()
        self.buttons.append({'rect': pygame.Rect(CONTROL_START_X + 90, button_y, 70, 35),
                            'text': '[Reset]', 'color': SECONDARY_COLOR, 'action': reset_algorithm})
        
        def reset_view():
            self.visualizer.reset_view()
            self.update_patient_positions()
        self.buttons.append({'rect': pygame.Rect(CONTROL_START_X + 170, button_y, 70, 35),
                            'text': '[View]', 'color': (100, 100, 150), 'action': reset_view})
        
        def go_to_config():
            self.running = False
        self.buttons.append({'rect': pygame.Rect(CONTROL_START_X + 10, button_y + 45, 70, 35),
                            'text': '[Conf]', 'color': (150, 100, 50), 'action': go_to_config})
        
        def save_route():
            if self.ga.best_solution:
                data = {'timestamp': datetime.now().isoformat(), 'generation': self.ga.generation,
                       'fitness': self.ga.best_fitness, 'total_distance_km': self.ga.best_distance,
                       'init_method': self.init_method.value, 'selection_method': self.selection_method.value,
                       'priority_metrics': self.ga.priority_metrics,
                       'route': [{'name': p.name, 'city': p.city_name, 'lat': p.lat, 'lon': p.lon,
                                 'priority': p.priority_score, 'urgency': p.urgency_level.display_name}
                                for p in self.ga.best_solution]}
                filename = f"mg_route_priority_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f" Rota salva em {filename}")
                print(f" Distancia total: {self.ga.best_distance:.1f} km")
        self.buttons.append({'rect': pygame.Rect(CONTROL_START_X + 90, button_y + 45, 70, 35),
                            'text': '[Save]', 'color': SUCCESS_COLOR, 'action': save_route})
        
        def toggle_distance():
            self.show_distance = not self.show_distance
        self.buttons.append({'rect': pygame.Rect(CONTROL_START_X + 170, button_y + 45, 70, 35),
                            'text': '[Dist]', 'color': (100, 100, 150), 'action': toggle_distance})
        
        def toggle_legend():
            self.show_legend = not self.show_legend
        self.buttons.append({'rect': pygame.Rect(CONTROL_START_X + 10, button_y + 90, 70, 35),
                            'text': '[Leg]', 'color': (100, 100, 150), 'action': toggle_legend})
    
    def draw_map_background(self):
        pygame.draw.rect(self.screen, (20, 40, 30), (0, 0, MAP_WIDTH, HEIGHT))
        grid_font_size = max(6, int(9 / max(0.5, self.visualizer.zoom)))
        font = pygame.font.SysFont('Arial', grid_font_size)
        for lat in range(int(MG_LAT_MIN), int(MG_LAT_MAX) + 1, 1):
            y = lat_lon_to_screen(lat, MG_LON_MIN, self.visualizer.zoom, 
                                  self.visualizer.offset_x, self.visualizer.offset_y)[1]
            if 0 <= y <= HEIGHT:
                line_width = max(1, int(1 / max(0.5, self.visualizer.zoom)))
                pygame.draw.line(self.screen, (40, 60, 50), (0, y), (MAP_WIDTH, y), line_width)
                lat_text = font.render(f"{abs(lat)}°S", True, (60, 80, 70))
                self.screen.blit(lat_text, (5, y - 5))
        for lon in range(int(MG_LON_MIN), int(MG_LON_MAX) + 1, 2):
            x = lat_lon_to_screen(MG_LAT_MIN, lon, self.visualizer.zoom, 
                                  self.visualizer.offset_x, self.visualizer.offset_y)[0]
            if 0 <= x <= MAP_WIDTH:
                line_width = max(1, int(1 / max(0.5, self.visualizer.zoom)))
                pygame.draw.line(self.screen, (40, 60, 50), (x, 0), (x, HEIGHT), line_width)
                lon_text = font.render(f"{abs(lon)}°O", True, (60, 80, 70))
                self.screen.blit(lon_text, (x + 2, 5))
        title_font_size = max(12, int(18 / max(0.8, self.visualizer.zoom)))
        title_font = pygame.font.SysFont('segoeui', title_font_size, bold=True)
        mg_title = title_font.render("MINAS GERAIS", True, (100, 150, 120))
        title_rect = mg_title.get_rect(center=(MAP_WIDTH // 2, 20))
        self.screen.blit(mg_title, title_rect)
        zoom_info = f"Zoom: {self.visualizer.zoom:.1f}x | Nivel: "
        if self.visualizer.detail_level == "low":
            zoom_info += "Visao Geral"
        elif self.visualizer.detail_level == "normal":
            zoom_info += "Cidades"
        elif self.visualizer.detail_level == "high":
            zoom_info += "Detalhes"
        else:
            zoom_info += "Ultra Detalhe"
        zoom_text = self.small_font.render(zoom_info, True, WARNING_COLOR)
        self.screen.blit(zoom_text, (15, HEIGHT - 80))
        method_info = self.small_font.render(f"Init: {self.init_method.value[:12]} | Sel: {self.selection_method.value[:12]}", True, SUCCESS_COLOR)
        self.screen.blit(method_info, (MAP_WIDTH - 280, HEIGHT - 25))
        priority_info = self.small_font.render("🔴 PRIORIDADE MAXIMA para EMERGENCIAS", True, COLOR_EMERGENCY)
        self.screen.blit(priority_info, (MAP_WIDTH - 280, HEIGHT - 45))
        unique_info = self.small_font.render("✅ Cidades unicas (sem repeticoes)", True, SUCCESS_COLOR)
        self.screen.blit(unique_info, (MAP_WIDTH - 280, HEIGHT - 65))
    
    def draw_info_panel(self):
        panel_rect = pygame.Rect(INFO_START_X, 0, INFO_PANEL_WIDTH, HEIGHT)
        pygame.draw.rect(self.screen, CARD_BG, panel_rect)
        pygame.draw.line(self.screen, PRIMARY_COLOR, (INFO_START_X, 0), (INFO_START_X, HEIGHT), 3)
        title = self.title_font.render("PACIENTE SELECIONADO", True, PRIMARY_COLOR)
        self.screen.blit(title, (INFO_START_X + 15, 15))
        if self.selected_patient:
            p = self.selected_patient
            card_y = 60
            pygame.draw.rect(self.screen, DARK_GRAY, (INFO_START_X + 10, card_y, INFO_PANEL_WIDTH - 20, 260), border_radius=10)
            name_text = self.subtitle_font.render(p.name, True, WHITE)
            self.screen.blit(name_text, (INFO_START_X + 20, card_y + 10))
            city_text = self.normal_font.render(f"Cidade: {p.city_name}", True, LIGHT_GRAY)
            self.screen.blit(city_text, (INFO_START_X + 20, card_y + 38))
            pygame.draw.line(self.screen, MEDIUM_GRAY, (INFO_START_X + 20, card_y + 58), (INFO_START_X + INFO_PANEL_WIDTH - 20, card_y + 58), 1)
            urgency_text = self.normal_font.render(f"Urgencia: {p.urgency_level.display_name}", True, p.get_color())
            self.screen.blit(urgency_text, (INFO_START_X + 20, card_y + 75))
            cat_text = self.normal_font.render(f"Categoria: {p.medical_category.display_name}", True, LIGHT_GRAY)
            self.screen.blit(cat_text, (INFO_START_X + 20, card_y + 100))
            priority_color = SUCCESS_COLOR if p.priority_score > 70 else WARNING_COLOR
            priority_text = self.normal_font.render(f"Prioridade: {p.priority_score:.0f}/100", True, priority_color)
            self.screen.blit(priority_text, (INFO_START_X + 20, card_y + 125))
            bar_rect = pygame.Rect(INFO_START_X + 20, card_y + 145, INFO_PANEL_WIDTH - 40, 8)
            pygame.draw.rect(self.screen, MEDIUM_GRAY, bar_rect, border_radius=4)
            fill_rect = pygame.Rect(INFO_START_X + 20, card_y + 145, (INFO_PANEL_WIDTH - 40) * p.priority_score / 100, 8)
            pygame.draw.rect(self.screen, priority_color, fill_rect, border_radius=4)
            time_text = self.normal_font.render(f"Tempo estimado: {p.estimated_service_time} min", True, LIGHT_GRAY)
            self.screen.blit(time_text, (INFO_START_X + 20, card_y + 168))
            coord_text = self.small_font.render(f"Lat: {abs(p.lat):.4f}°S, Lon: {abs(p.lon):.4f}°O", True, LIGHT_GRAY)
            self.screen.blit(coord_text, (INFO_START_X + 20, card_y + 193))
            addr_text = self.small_font.render(f"End: {p.address}", True, LIGHT_GRAY)
            self.screen.blit(addr_text, (INFO_START_X + 20, card_y + 215))
            phone_text = self.small_font.render(f"Tel: {p.phone}", True, LIGHT_GRAY)
            self.screen.blit(phone_text, (INFO_START_X + 20, card_y + 237))
        else:
            no_select = self.normal_font.render("Clique em um paciente", True, MEDIUM_GRAY)
            self.screen.blit(no_select, (INFO_START_X + 20, 100))
            no_select2 = self.small_font.render("para ver detalhes", True, MEDIUM_GRAY)
            self.screen.blit(no_select2, (INFO_START_X + 20, 125))
    
    def draw_control_panel(self):
        panel_rect = pygame.Rect(CONTROL_START_X, 0, CONTROL_PANEL_WIDTH, HEIGHT)
        pygame.draw.rect(self.screen, CARD_BG, panel_rect)
        pygame.draw.line(self.screen, PRIMARY_COLOR, (CONTROL_START_X, 0), (CONTROL_START_X, HEIGHT), 2)
        title = self.title_font.render("CONTROLES", True, PRIMARY_COLOR)
        self.screen.blit(title, (CONTROL_START_X + 15, 15))
        for button in self.buttons:
            if button['text'] in ['[Pause]', '[Start]']:
                button['text'] = '[Pause]' if not self.paused else '[Start]'
            mouse_pos = pygame.mouse.get_pos()
            color = button['color']
            if button['rect'].collidepoint(mouse_pos):
                color = tuple(min(255, c + 30) for c in color)
            pygame.draw.rect(self.screen, color, button['rect'], border_radius=6)
            text = self.normal_font.render(button['text'], True, WHITE)
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)
        stats_y = 140
        stats_title = self.subtitle_font.render("EVOLUCAO", True, LIGHT_GRAY)
        self.screen.blit(stats_title, (CONTROL_START_X + 15, stats_y))
        distance_text = f"Distancia: {self.ga.best_distance:.1f} km"
        fitness_text = f"Fitness: {self.ga.best_fitness:.1f} (menor=melhor)"
        stats = [f"Geracao: {self.ga.generation}/{self.num_generations}",
                fitness_text, distance_text, f"Populacao: {len(self.ga.population)}"]
        for i, stat in enumerate(stats):
            color = SUCCESS_COLOR if "km" in stat and self.show_distance else WHITE
            text = self.small_font.render(stat, True, color)
            self.screen.blit(text, (CONTROL_START_X + 15, stats_y + 30 + i * 22))
        priority_y = stats_y + 130
        priority_title = self.subtitle_font.render("PRIORIDADE", True, LIGHT_GRAY)
        self.screen.blit(priority_title, (CONTROL_START_X + 15, priority_y))
        priority_stats = [
            f"Emerg. inicio: {self.ga.priority_metrics.get('emergency_in_first_20', False) and 'SIM' or 'NAO'}",
            f"Cobertura: {self.ga.priority_metrics.get('priority_coverage', 0) * 100:.0f}%",
            f"Total Emerg: {self.ga.priority_metrics.get('total_emergencies', 0)}"
        ]
        for i, stat in enumerate(priority_stats):
            text = self.small_font.render(stat, True, WARNING_COLOR)
            self.screen.blit(text, (CONTROL_START_X + 15, priority_y + 30 + i * 22))
        if self.show_legend:
            legend_y = HEIGHT - 200
            legend_title = self.small_font.render("URGENCIA", True, LIGHT_GRAY)
            self.screen.blit(legend_title, (CONTROL_START_X + 15, legend_y))
            legend_items = [(COLOR_EMERGENCY, "Emerg (Prior Max)"), (COLOR_HIGH, "Alta"), 
                           (COLOR_MEDIUM, "Media"), (COLOR_LOW, "Baixa"), (COLOR_ROUTINE, "Rotina")]
            for i, (color, label) in enumerate(legend_items):
                pygame.draw.circle(self.screen, color, (CONTROL_START_X + 25, legend_y + 22 + i * 18), 6)
                text = self.small_font.render(label, True, LIGHT_GRAY)
                self.screen.blit(text, (CONTROL_START_X + 45, legend_y + 19 + i * 18))
    
    def draw_progress_panel(self):
        panel_rect = pygame.Rect(MAP_WIDTH - 280, 15, 265, 110)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, PRIMARY_COLOR, panel_rect, 2, border_radius=10)
        title = self.small_font.render("EVOLUCAO DA ROTA", True, PRIMARY_COLOR)
        self.screen.blit(title, (panel_rect.x + 10, panel_rect.y + 8))
        if len(self.ga.fitness_history) > 1:
            history = self.ga.fitness_history[-50:]
            if history:
                max_fitness, min_fitness = max(history), min(history)
                range_fitness = max_fitness - min_fitness or 1
                graph_rect = pygame.Rect(panel_rect.x + 10, panel_rect.y + 28, 245, 35)
                fit_label = self.small_font.render("Fitness", True, LIGHT_GRAY)
                self.screen.blit(fit_label, (graph_rect.x, graph_rect.y - 12))
                for i in range(1, len(history)):
                    x1 = graph_rect.x + (i-1) * graph_rect.width / (len(history) - 1)
                    y1 = graph_rect.y + graph_rect.height - (history[i-1] - min_fitness) * graph_rect.height / range_fitness
                    x2 = graph_rect.x + i * graph_rect.width / (len(history) - 1)
                    y2 = graph_rect.y + graph_rect.height - (history[i] - min_fitness) * graph_rect.height / range_fitness
                    color = SUCCESS_COLOR if history[i] < history[i-1] else PRIMARY_COLOR
                    pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 2)
            if len(self.ga.distance_history) > 1:
                dist_history = self.ga.distance_history[-50:]
                max_dist, min_dist = max(dist_history), min(dist_history)
                range_dist = max_dist - min_dist or 1
                graph_rect = pygame.Rect(panel_rect.x + 10, panel_rect.y + 70, 245, 30)
                dist_label = self.small_font.render("Distancia (km)", True, LIGHT_GRAY)
                self.screen.blit(dist_label, (graph_rect.x, graph_rect.y - 12))
                for i in range(1, len(dist_history)):
                    x1 = graph_rect.x + (i-1) * graph_rect.width / (len(dist_history) - 1)
                    y1 = graph_rect.y + graph_rect.height - (dist_history[i-1] - min_dist) * graph_rect.height / range_dist
                    x2 = graph_rect.x + i * graph_rect.width / (len(dist_history) - 1)
                    y2 = graph_rect.y + graph_rect.height - (dist_history[i] - min_dist) * graph_rect.height / range_dist
                    color = SUCCESS_COLOR if dist_history[i] < dist_history[i-1] else (255, 165, 0)
                    pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 2)
    
    def draw_distance_overlay(self):
        if not self.show_distance or not self.ga.best_solution:
            return
        overlay_rect = pygame.Rect(MAP_WIDTH - 250, HEIGHT - 100, 240, 85)
        pygame.draw.rect(self.screen, (0, 0, 0, 200), overlay_rect, border_radius=8)
        pygame.draw.rect(self.screen, SUCCESS_COLOR, overlay_rect, 2, border_radius=8)
        dist_title = self.small_font.render("DISTANCIA TOTAL", True, SUCCESS_COLOR)
        self.screen.blit(dist_title, (overlay_rect.x + 10, overlay_rect.y + 8))
        dist_value = self.big_font.render(f"{self.ga.best_distance:.1f} km", True, WHITE)
        self.screen.blit(dist_value, (overlay_rect.x + 10, overlay_rect.y + 28))
        cities_count = len(set(p.city_name for p in self.ga.best_solution))
        cities_text = self.small_font.render(f"{cities_count} cidades unicas", True, LIGHT_GRAY)
        self.screen.blit(cities_text, (overlay_rect.x + 10, overlay_rect.y + 55))
        fitness_text = self.small_font.render(f"Fitness: {self.ga.best_fitness:.1f}", True, WARNING_COLOR)
        self.screen.blit(fitness_text, (overlay_rect.x + 10, overlay_rect.y + 70))
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_l:
                    self.show_legend = not self.show_legend
                elif event.key == pygame.K_d:
                    self.show_distance = not self.show_distance
                elif event.key == pygame.K_c:
                    self.running = False
                elif event.key == pygame.K_r:
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.visualizer.reset_view()
                        self.update_patient_positions()
                    else:
                        self.initialize_algorithm()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in self.buttons:
                    if button['rect'].collidepoint(event.pos):
                        button['action']()
                if event.pos[0] < MAP_WIDTH:
                    self.select_patient_at_position(event.pos)
            self.visualizer.handle_zoom_event(event)
            self.visualizer.handle_pan_event(event)
            if event.type in [pygame.MOUSEWHEEL, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION]:
                self.update_patient_positions()
    
    def select_patient_at_position(self, pos):
        min_dist = 25 / self.visualizer.zoom
        for patient in self.patients:
            if patient.screen_pos:
                dx = patient.screen_pos[0] - pos[0]
                dy = patient.screen_pos[1] - pos[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    self.selected_patient = patient
                    min_dist = dist
                    self.visualizer.add_particle(patient.screen_pos[0], patient.screen_pos[1], patient.get_color())
    
    def update(self):
        if not self.paused and self.ga.generation < self.num_generations:
            evolution_data = self.ga.evolve_generation()
            if evolution_data['improved'] and self.ga.best_solution:
                for patient in self.ga.best_solution:
                    patient.update_screen_pos(self.visualizer.zoom, self.visualizer.offset_x, self.visualizer.offset_y)
                for i in range(min(5, len(self.ga.best_solution) - 1)):
                    p1, p2 = self.ga.best_solution[i], self.ga.best_solution[i + 1]
                    self.visualizer.add_connection_highlight(p1, p2)
        self.visualizer.update_animation()
    
    def draw(self):
        self.draw_map_background()
        if self.ga.best_solution:
            self.visualizer.draw_route(self.ga.best_solution)
        for patient in self.patients:
            self.visualizer.draw_patient(patient, self.selected_patient == patient)
        self.draw_info_panel()
        self.draw_control_panel()
        self.draw_progress_panel()
        self.draw_distance_overlay()
        status_text = "[RUNNING]" if not self.paused else "[PAUSED]"
        status_color = SUCCESS_COLOR if not self.paused else WARNING_COLOR
        status_surf = self.big_font.render(status_text, True, status_color)
        self.screen.blit(status_surf, (15, 15))
        fps_text = self.small_font.render(f"FPS: {int(self.clock.get_fps())}", True, LIGHT_GRAY)
        self.screen.blit(fps_text, (15, HEIGHT - 25))
        help_text = self.small_font.render("Scroll: Zoom | Drag: Pan | R: Reset | C: Config | L: Legenda", True, MEDIUM_GRAY)
        self.screen.blit(help_text, (15, HEIGHT - 45))
        progress = self.ga.generation / self.num_generations
        bar_rect = pygame.Rect(15, HEIGHT - 65, 300, 15)
        pygame.draw.rect(self.screen, DARK_GRAY, bar_rect, border_radius=7)
        pygame.draw.rect(self.screen, PRIMARY_COLOR, (bar_rect.x, bar_rect.y, bar_rect.width * progress, bar_rect.height), border_radius=7)
        progress_text = self.small_font.render(f"Progresso: {self.ga.generation}/{self.num_generations}", True, LIGHT_GRAY)
        self.screen.blit(progress_text, (15, HEIGHT - 85))
    
    def run(self):
        while self.running and self.ga.generation < self.num_generations:
            self.handle_events()
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        
        if self.ga.best_solution:
            print("\n" + "="*70)
            print(" RESULTADO FINAL - ROTEAMENTO PRIORIZADO")
            print("="*70)
            print(f" Distancia total: {self.ga.best_distance:.1f} km")
            print(f" Melhor fitness: {self.ga.best_fitness:.2f} (menor = melhor)")
            print(f" Geracoes executadas: {self.ga.generation}/{self.num_generations}")
            print(f" Metodo de inicializacao: {self.init_method.value}")
            print(f" Metodo de selecao: {self.selection_method.value}")
            print(f" Pacientes atendidos: {len(self.ga.best_solution)}")
            print(f" Cidades unicas visitadas: {len(set(p.city_name for p in self.ga.best_solution))}")
            
            # Métricas de prioridade
            print("\n METRICAS DE PRIORIZACAO:")
            print(f"   Emergencias na rota: {self.ga.priority_metrics.get('total_emergencies', 0)}")
            print(f"   Emergencias no inicio: {self.ga.priority_metrics.get('emergency_in_first_20', False) and 'SIM ✅' or 'NAO ❌'}")
            print(f"   Cobertura de prioridade: {self.ga.priority_metrics.get('priority_coverage', 0) * 100:.1f}%")
            
            print("\n" + "="*70)
            print(" ORDEM DE ATENDIMENTO (TODAS AS CIDADES)")
            print("="*70)
            print()
            
            # Mostrar TODAS as cidades sem truncamento
            for i, patient in enumerate(self.ga.best_solution):
                # Formatar com indicador de prioridade
                priority_indicator = "🔴" if patient.urgency_level == UrgencyLevel.EMERGENCY else "🟠" if patient.urgency_level == UrgencyLevel.HIGH else "🟡" if patient.urgency_level == UrgencyLevel.MEDIUM else "🟢" if patient.urgency_level == UrgencyLevel.LOW else "⚪"
                print(f"  {i+1:3d}. {priority_indicator} {patient.city_name:30s} -> {patient.name:20s} | {patient.urgency_level.display_name:10s} | Prio: {patient.priority_score:3.0f}")
                
                # A cada 10 cidades, mostrar uma linha em branco
                if (i + 1) % 10 == 0 and i + 1 < len(self.ga.best_solution):
                    print()
            
            print("\n" + "="*70)
            print(" ESTATISTICAS ADICIONAIS")
            print("="*70)
            
            # Contar por região
            region_count = {}
            region_names = {
                "Norte": "Norte de Minas",
                "Sul": "Sul de Minas",
                "Triangulo": "Triangulo Mineiro",
                "Rio Doce": "Vale do Rio Doce",
                "Metropolitana": "Regiao Metropolitana",
                "Mata": "Zona da Mata",
                "Centro-Oeste": "Centro-Oeste",
                "Jequitinhonha": "Jequitinhonha",
                "Noroeste": "Noroeste"
            }
            
            for patient in self.ga.best_solution:
                region = CITIES_MG[patient.city_name]["region"]
                region_count[region] = region_count.get(region, 0) + 1
            
            print("\n Cidades por regiao:")
            for region, count in sorted(region_count.items(), key=lambda x: -x[1]):
                region_display = region_names.get(region, region)
                bar = "█" * min(30, count * 2)
                print(f"   {region_display:25s}: {count:2d} cidades {bar}")
            
            # Contar por urgência
            urgency_count = {}
            for patient in self.ga.best_solution:
                urgency_count[patient.urgency_level.display_name] = urgency_count.get(patient.urgency_level.display_name, 0) + 1
            
            print("\n Atendimentos por urgencia:")
            urgency_order = ["EMERGENCIA", "ALTA", "MEDIA", "BAIXA", "ROTINA"]
            for urgency in urgency_order:
                if urgency in urgency_count:
                    count = urgency_count[urgency]
                    bar = "█" * min(30, count * 2)
                    print(f"   {urgency:12s}: {count:2d} pacientes {bar}")
            
            # Calcular tempo total estimado
            tempo_total_horas = self.ga.best_distance / 60
            tempo_total_minutos = tempo_total_horas * 60
            
            print(f"\n Tempo estimado de viagem:")
            print(f"   → {tempo_total_horas:.1f} horas (a 60 km/h)")
            print(f"   → {tempo_total_minutos:.0f} minutos")
            print(f"   → Distancia media entre paradas: {self.ga.best_distance / len(self.ga.best_solution):.1f} km")
            
            # Mostrar as 3 primeiras e últimas cidades para verificar priorização
            print("\n VERIFICACAO DE PRIORIZACAO:")
            print("   Primeiros atendimentos (deveriam ser emergencias/alta prioridade):")
            for i, patient in enumerate(self.ga.best_solution[:5]):
                indicator = "🔴" if patient.urgency_level == UrgencyLevel.EMERGENCY else "🟠" if patient.urgency_level == UrgencyLevel.HIGH else "🟡"
                print(f"     {i+1}. {indicator} {patient.city_name} - {patient.urgency_level.display_name}")
            
            print("   Ultimos atendimentos (deveriam ser baixa prioridade/rotina):")
            for i, patient in enumerate(self.ga.best_solution[-5:]):
                index = len(self.ga.best_solution) - 5 + i + 1
                indicator = "🟢" if patient.urgency_level == UrgencyLevel.LOW else "⚪" if patient.urgency_level == UrgencyLevel.ROUTINE else "🟡"
                print(f"     {index}. {indicator} {patient.city_name} - {patient.urgency_level.display_name}")
        
        pygame.quit()
        return not self.running

# ==================== PONTO DE ENTRADA ====================

if __name__ == "__main__":
    print("=" * 70)
    print(" SISTEMA DE ROTEAMENTO MEDICO - SAUDE DA MULHER")
    print(" MINAS GERAIS - PRIORIZACAO DE ATENDIMENTOS URGENTES")
    print("=" * 70)
    print("\n OBJETIVO: Atender primeiro pacientes com maior urgencia")
    print("   🔴 EMERGENCIA → Prioridade MAXIMA (atendidas primeiro)")
    print("   🟠 ALTA PRIORIDADE → Segundo nivel de prioridade")
    print("   🟡 MEDIA PRIORIDADE → Prioridade moderada")
    print("   🟢 BAIXA PRIORIDADE → Atendidas depois")
    print("   ⚪ ROTINA → Atendidas por ultimo")
    print("\n O algoritmo vai otimizar a rota para que pacientes")
    print(" mais urgentes sejam atendidos o mais rapido possivel!\n")
    
    while True:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Configuracao - Roteamento Medico MG (Priorizacao)")
        
        config = ConfigScreen(screen)
        params = config.run()
        
        pygame.quit()
        
        system = HealthcareRoutingSystem(
            population_size=params['population_size'],
            num_generations=params['num_generations'],
            mutation_rate=params['mutation_rate'],
            num_patients=params['num_patients'],
            init_method=params['init_method'],
            selection_method=params['selection_method']
        )
        
        should_restart = system.run()
        
        if not should_restart:
            break
    
    sys.exit()