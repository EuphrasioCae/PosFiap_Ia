# -*- coding: utf-8 -*-
"""
Sistema de Roteamento Médico para Saúde da Mulher
Visualização em Tempo Real com Métricas de Distância
"""

import pygame
from pygame.locals import *
import random
import math
import copy
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from enum import Enum
import sys
from collections import deque

# Configurar encoding para Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# ==================== CONSTANTES ====================

WIDTH, HEIGHT = 1400, 800
NODE_RADIUS = 10
FPS = 60

# Dimensões dos painéis
MAP_WIDTH = WIDTH - 450
INFO_PANEL_WIDTH = 250
CONTROL_PANEL_WIDTH = 200
INFO_START_X = MAP_WIDTH
CONTROL_START_X = MAP_WIDTH + INFO_PANEL_WIDTH

# Cores
WHITE = (255, 255, 255)
BLACK = (10, 10, 20)
DARK_GRAY = (30, 30, 40)
MEDIUM_GRAY = (60, 60, 70)
LIGHT_GRAY = (180, 180, 190)

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

# ==================== ENUMS ====================

class UrgencyLevel(Enum):
    EMERGENCY = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    ROUTINE = 1
    
    @property
    def display_name(self):
        names = {
            5: "EMERGENCIA",
            4: "ALTA",
            3: "MEDIA",
            2: "BAIXA",
            1: "ROTINA"
        }
        return names[self.value]
    
    @property
    def icon(self):
        icons = {
            5: "[E]",
            4: "[A]",
            3: "[M]",
            2: "[B]",
            1: "[R]"
        }
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
            1: "Gestacao",
            2: "Pos-parto",
            3: "Hormonal",
            4: "Emergencia",
            5: "Pre-natal",
            6: "Violencia",
            7: "Area Remota",
            8: "Check-up"
        }
        return names[self.value]
    
    @property
    def icon(self):
        icons = {
            1: "[G]",
            2: "[P]",
            3: "[H]",
            4: "[E]",
            5: "[N]",
            6: "[V]",
            7: "[R]",
            8: "[C]"
        }
        return icons[self.value]

# ==================== CLASSES DE DADOS ====================

class Patient:
    """Paciente com todas as informações para roteamento prioritário"""
    def __init__(self, id, name, location, urgency_level, medical_category, 
                 estimated_service_time=30, address="", phone=""):
        self.id = id
        self.name = name
        self.location = location
        self.urgency_level = urgency_level
        self.medical_category = medical_category
        self.estimated_service_time = estimated_service_time
        self.address = address
        self.phone = phone
        self.priority_score = self.calculate_priority_score()
        self.arrival_time = None
        
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
        """Calcula pontuação de prioridade (0-100)"""
        base_score = self.urgency_level.value * 15
        
        # Bonus por categoria especial
        bonuses = {
            MedicalCategory.DOMESTIC_VIOLENCE: 25,
            MedicalCategory.POSTPARTUM: 20,
            MedicalCategory.EMERGENCY: 30,
            MedicalCategory.REMOTE_AREA: 10
        }
        
        base_score += bonuses.get(self.medical_category, 0)
        return min(100, base_score)
    
    def __eq__(self, other):
        if not isinstance(other, Patient):
            return False
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)
    
    def __repr__(self):
        return f"{self.name} [{self.urgency_level.display_name}]"

class RouteEvolution:
    """Registra a evolução da rota para animação"""
    def __init__(self):
        self.history = deque(maxlen=50)
        self.improvements = []
        self.last_fitness = float('inf')
        self.last_distance = float('inf')
        
    def add_snapshot(self, route, fitness, distance, generation):
        self.history.append({
            'route': copy.deepcopy(route),
            'fitness': fitness,
            'distance': distance,
            'generation': generation
        })
        
        if fitness < self.last_fitness * 0.95:
            self.improvements.append({
                'generation': generation,
                'fitness': fitness,
                'distance': distance,
                'fade': 30
            })
        
        self.last_fitness = fitness
        self.last_distance = distance
        
    def update_improvements(self):
        self.improvements = [imp for imp in self.improvements if imp['fade'] > 0]
        for imp in self.improvements:
            imp['fade'] -= 1

# ==================== ALGORITMO GENÉTICO ====================

class PriorityGeneticAlgorithm:
    """Algoritmo Genético com foco em prioridade de pacientes"""
    
    def __init__(self, patients, population_size=150, mutation_rate=0.35, 
                 crossover_rate=0.8, elitism_count=5):
        self.patients = patients
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        
        self.population = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_distance = float('inf')
        self.fitness_history = []
        self.distance_history = []
        self.generation = 0
        self.convergence_count = 0
        
        # Métricas de prioridade
        self.priority_metrics = {
            'emergency_first': 0,
            'avg_priority_position': 0,
            'priority_coverage': 0
        }
        
    def calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_total_distance(self, route):
        """Calcula a distância total de uma rota"""
        if not route or len(route) < 2:
            return 0
        
        total = 0
        for i in range(len(route)):
            p1 = route[i].location
            p2 = route[(i + 1) % len(route)].location  # Fecha o ciclo
            total += self.calculate_distance(p1, p2)
        return total
    
    def calculate_fitness(self, route):
        if not route:
            return float('inf')
        
        total_distance = 0
        priority_sum = 0
        position_penalty = 0
        
        n = len(route)
        
        for i, patient in enumerate(route):
            priority_sum += patient.priority_score
            
            if patient.urgency_level in [UrgencyLevel.EMERGENCY, UrgencyLevel.HIGH]:
                position_factor = (n - i) / n
                position_penalty += patient.priority_score * position_factor * 2
            
            if i > 0:
                dist = self.calculate_distance(route[i-1].location, patient.location)
                total_distance += dist
        
        distance_cost = total_distance / 1000
        priority_benefit = priority_sum / 10
        fitness = distance_cost - priority_benefit + position_penalty
        
        return max(0.01, fitness)
    
    def calculate_priority_metrics(self, route):
        n = len(route)
        if n == 0:
            return
        
        high_priority_positions = []
        emergency_found = False
        
        for i, patient in enumerate(route):
            if patient.urgency_level in [UrgencyLevel.EMERGENCY, UrgencyLevel.HIGH]:
                high_priority_positions.append(i)
                if patient.urgency_level == UrgencyLevel.EMERGENCY and i < n * 0.2:
                    emergency_found = True
        
        if high_priority_positions:
            self.priority_metrics['avg_priority_position'] = sum(high_priority_positions) / len(high_priority_positions) / n
        else:
            self.priority_metrics['avg_priority_position'] = 1
        
        self.priority_metrics['emergency_first'] = 1 if emergency_found else 0
        
        high_priority_in_first_30 = sum(1 for i, p in enumerate(route) 
                                       if i < n * 0.3 and p.urgency_level.value >= 4)
        total_high_priority = sum(1 for p in route if p.urgency_level.value >= 4)
        
        if total_high_priority > 0:
            self.priority_metrics['priority_coverage'] = high_priority_in_first_30 / total_high_priority
        else:
            self.priority_metrics['priority_coverage'] = 1
    
    def initialize_population(self):
        self.population = []
        
        for _ in range(self.population_size):
            sorted_patients = sorted(self.patients, 
                                    key=lambda p: (-p.priority_score, -p.urgency_level.value))
            
            strategy = random.random()
            
            if strategy < 0.6:
                route = sorted_patients.copy()
                swap_count = random.randint(1, max(1, len(route) // 10))
                for _ in range(swap_count):
                    i, j = random.sample(range(len(route)), 2)
                    if abs(route[i].priority_score - route[j].priority_score) < 20:
                        route[i], route[j] = route[j], route[i]
                        
            elif strategy < 0.8:
                route = []
                remaining = sorted_patients.copy()
                current_pos = (random.randint(100, MAP_WIDTH - 100), 
                              random.randint(100, HEIGHT - 100))
                
                while remaining:
                    nearest = min(remaining, 
                                key=lambda p: self.calculate_distance(current_pos, p.location))
                    route.append(nearest)
                    current_pos = nearest.location
                    remaining.remove(nearest)
                    
            else:
                route = random.sample(self.patients, len(self.patients))
                for i in range(0, len(route), 3):
                    segment = route[i:min(i+3, len(route))]
                    segment.sort(key=lambda p: -p.priority_score)
                    route[i:min(i+3, len(route))] = segment
            
            self.population.append(route)
    
    def select_parent(self, fitness_values):
        tournament_size = 5
        indices = random.sample(range(len(self.population)), tournament_size)
        best_idx = min(indices, key=lambda i: fitness_values[i])
        return copy.deepcopy(self.population[best_idx])
    
    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)
        
        n = len(parent1)
        high_priority_ids = set()
        
        for i, p in enumerate(parent1):
            if p.priority_score > 60:
                high_priority_ids.add(p.id)
        
        child = [None] * n
        
        for i, patient in enumerate(parent1):
            if patient.id in high_priority_ids:
                child[i] = patient
        
        remaining = []
        for patient in parent2:
            if patient.id not in high_priority_ids:
                remaining.append(patient)
        
        pos = 0
        for i in range(n):
            if child[i] is None:
                child[i] = remaining[pos]
                pos += 1
        
        return child
    
    def mutate(self, route):
        if random.random() > self.mutation_rate:
            return copy.deepcopy(route)
        
        mutated = copy.deepcopy(route)
        n = len(mutated)
        
        if n < 2:
            return mutated
        
        mutation_type = random.choice(['swap_priority', 'shift_priority', 'invert', 'local_sort'])
        
        if mutation_type == 'swap_priority' and n >= 2:
            high_indices = [i for i, p in enumerate(mutated) if p.priority_score > 60]
            low_indices = [i for i, p in enumerate(mutated) if p.priority_score < 40]
            
            if high_indices and low_indices:
                i = random.choice(high_indices)
                j = random.choice(low_indices)
                if i > j:
                    mutated[i], mutated[j] = mutated[j], mutated[i]
                    
        elif mutation_type == 'shift_priority' and n >= 2:
            high_indices = [i for i, p in enumerate(mutated) if p.priority_score > 60]
            if high_indices:
                idx = random.choice(high_indices)
                if idx > 0:
                    patient = mutated.pop(idx)
                    new_pos = max(0, idx - random.randint(1, min(3, idx)))
                    mutated.insert(new_pos, patient)
                    
        elif mutation_type == 'invert' and n >= 3:
            start = random.randint(0, n - 3)
            end = start + random.randint(2, min(4, n - start))
            mutated[start:end] = reversed(mutated[start:end])
            
        elif mutation_type == 'local_sort' and n >= 3:
            start = random.randint(0, max(0, n - 4))
            end = min(n, start + random.randint(3, 5))
            segment = mutated[start:end]
            segment.sort(key=lambda p: -p.priority_score)
            mutated[start:end] = segment
        
        return mutated
    
    def evolve_generation(self):
        fitness_values = [self.calculate_fitness(route) for route in self.population]
        distance_values = [self.calculate_total_distance(route) for route in self.population]
        
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
            self.convergence_count = 0
        else:
            self.convergence_count += 1
        
        self.fitness_history.append(self.best_fitness)
        self.distance_history.append(self.best_distance)
        
        # Elitismo
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
        
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_distance': self.best_distance,
            'avg_fitness': sum(fitness_values) / len(fitness_values),
            'improved': improved,
            'priority_metrics': self.priority_metrics.copy()
        }

# ==================== VISUALIZADOR ====================

class RealTimeVisualizer:
    def __init__(self, screen):
        self.screen = screen
        self.animation_offset = 0
        self.pulse_alpha = 0
        self.pulse_direction = 1
        self.particles = []
        self.connection_fade = {}
        
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
        self.particles.append({
            'x': x, 'y': y,
            'vx': random.uniform(-2, 2),
            'vy': random.uniform(-2, 2),
            'color': color,
            'life': 30
        })
    
    def draw_route(self, route):
        if not route or len(route) < 2:
            return
        
        for i in range(len(route) - 1):
            p1 = route[i]
            p2 = route[i + 1]
            base_color = p2.get_color()
            key = (p1.id, p2.id)
            
            if key in self.connection_fade:
                color = (255, 255, 100)
                width = 4
            else:
                color = base_color
                width = 2 + int(p2.priority_score / 50)
            
            pygame.draw.line(self.screen, color, p1.location, p2.location, width)
            
            if p2.urgency_level == UrgencyLevel.EMERGENCY:
                pulse_intensity = int(100 + 155 * self.pulse_alpha)
                pulse_color = (255, pulse_intensity, pulse_intensity)
                pygame.draw.line(self.screen, pulse_color, p1.location, p2.location, width + 2)
        
        # Desenhar linha de retorno (ciclo)
        if len(route) > 2:
            p1 = route[-1]
            p2 = route[0]
            pygame.draw.line(self.screen, (80, 80, 100), p1.location, p2.location, 1)
    
    def draw_patient(self, patient, is_selected=False, show_details=True):
        pos = patient.location
        radius = NODE_RADIUS + (2 if patient.urgency_level == UrgencyLevel.EMERGENCY else 0)
        
        if patient.urgency_level == UrgencyLevel.EMERGENCY:
            pulse_radius = radius + int(3 * self.pulse_alpha)
            pygame.draw.circle(self.screen, COLOR_EMERGENCY, pos, pulse_radius, 2)
        
        color = patient.get_color()
        pygame.draw.circle(self.screen, color, pos, radius)
        pygame.draw.circle(self.screen, WHITE, pos, radius, 2)
        
        if is_selected:
            pygame.draw.circle(self.screen, (255, 255, 100), pos, radius + 3, 3)
        
        # Icone simples
        font = pygame.font.SysFont('segoeui', 12)
        icon = font.render(patient.urgency_level.icon, True, WHITE)
        icon_rect = icon.get_rect(center=pos)
        self.screen.blit(icon, icon_rect)
        
        # Nome
        if show_details:
            name_font = pygame.font.SysFont('Arial', 9)
            name_text = name_font.render(patient.name[:12], True, WHITE)
            name_rect = name_text.get_rect(center=(pos[0], pos[1] - radius - 5))
            self.screen.blit(name_text, name_rect)
        
        # Partículas para emergências
        if patient.urgency_level == UrgencyLevel.EMERGENCY and random.random() < 0.1:
            self.add_particle(pos[0] + random.randint(-5, 5), 
                            pos[1] + random.randint(-5, 5), COLOR_EMERGENCY)
        
        for particle in self.particles:
            pygame.draw.circle(self.screen, particle['color'], 
                             (int(particle['x']), int(particle['y'])), 2)

# ==================== SISTEMA PRINCIPAL ====================

class HealthcareRoutingSystem:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Sistema de Roteamento Medico - Saude da Mulher")
        self.clock = pygame.time.Clock()
        
        # Fontes
        self.title_font = pygame.font.SysFont('segoeui', 22, bold=True)
        self.subtitle_font = pygame.font.SysFont('segoeui', 16, bold=True)
        self.normal_font = pygame.font.SysFont('segoeui', 13)
        self.small_font = pygame.font.SysFont('segoeui', 11)
        self.big_font = pygame.font.SysFont('segoeui', 28, bold=True)
        
        # Estado
        self.running = True
        self.paused = False
        self.selected_patient = None
        self.show_distance = True
        
        # Componentes
        self.visualizer = RealTimeVisualizer(self.screen)
        
        # Dados
        self.patients = self.generate_patients(25)
        self.ga = PriorityGeneticAlgorithm(self.patients, population_size=150, mutation_rate=0.35)
        self.ga.initialize_population()
        
        # Evolução
        self.route_evolution = RouteEvolution()
        
        # UI
        self.setup_ui()
        
    def generate_patients(self, n):
        names = [
            "Ana Silva", "Maria Santos", "Carla Oliveira", "Patricia Souza",
            "Fernanda Lima", "Juliana Costa", "Renata Ferreira", "Beatriz Almeida",
            "Luciana Rodrigues", "Marcia Pereira", "Tatiana Gomes", "Simone Carvalho",
            "Andreia Ribeiro", "Camila Dias", "Vanessa Nunes", "Roberta Monteiro"
        ]
        
        categories = list(MedicalCategory)
        patients = []
        
        # Clusters para simular regiões
        clusters = [
            (200, 200), (400, 500), (600, 300), (350, 700), (700, 600),
            (500, 200), (300, 400), (650, 450), (450, 650), (550, 350)
        ]
        
        for i in range(n):
            cluster = clusters[i % len(clusters)]
            x = cluster[0] + random.randint(-80, 80)
            y = cluster[1] + random.randint(-80, 80)
            x = max(50, min(MAP_WIDTH - 50, x))
            y = max(50, min(HEIGHT - 100, y))
            
            # Distribuição realista de categorias
            weights = [0.20, 0.12, 0.15, 0.05, 0.18, 0.08, 0.12, 0.10]
            category = random.choices(categories, weights=weights)[0]
            
            # Urgência baseada na categoria
            if category == MedicalCategory.EMERGENCY:
                urgency = UrgencyLevel.EMERGENCY
            elif category == MedicalCategory.DOMESTIC_VIOLENCE:
                urgency = UrgencyLevel.HIGH
            elif category == MedicalCategory.POSTPARTUM:
                urgency = random.choices([UrgencyLevel.HIGH, UrgencyLevel.MEDIUM], weights=[0.6, 0.4])[0]
            else:
                urgency = random.choices(list(UrgencyLevel), 
                                       weights=[0.05, 0.10, 0.20, 0.30, 0.35])[0]
            
            patient = Patient(
                id=i,
                name=names[i % len(names)],
                location=(x, y),
                urgency_level=urgency,
                medical_category=category,
                estimated_service_time=random.randint(20, 60),
                address=f"Rua {i+1}, {random.randint(100, 1000)}",
                phone=f"(11) 9{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
            )
            
            patients.append(patient)
        
        return patients
    
    def setup_ui(self):
        button_y = 20
        self.buttons = []
        
        def toggle_pause():
            self.paused = not self.paused
        self.buttons.append({
            'rect': pygame.Rect(CONTROL_START_X + 15, button_y, 80, 35),
            'text': '[Pause]' if not self.paused else '[Start]',
            'color': PRIMARY_COLOR,
            'action': toggle_pause
        })
        
        def reset_algorithm():
            self.ga = PriorityGeneticAlgorithm(self.patients, population_size=150)
            self.ga.initialize_population()
            self.paused = False
            self.route_evolution = RouteEvolution()
        self.buttons.append({
            'rect': pygame.Rect(CONTROL_START_X + 105, button_y, 80, 35),
            'text': '[Reset]',
            'color': SECONDARY_COLOR,
            'action': reset_algorithm
        })
        
        def save_route():
            if self.ga.best_solution:
                data = {
                    'timestamp': datetime.now().isoformat(),
                    'generation': self.ga.generation,
                    'fitness': self.ga.best_fitness,
                    'total_distance': self.ga.best_distance,
                    'route': [{'name': p.name, 'priority': p.priority_score, 
                              'urgency': p.urgency_level.display_name,
                              'location': p.location} 
                             for p in self.ga.best_solution]
                }
                filename = f"route_{datetime.now().strftime('%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"Rota salva em {filename} - Distancia: {self.ga.best_distance:.1f} pixels")
        self.buttons.append({
            'rect': pygame.Rect(CONTROL_START_X + 15, button_y + 45, 80, 35),
            'text': '[Save]',
            'color': SUCCESS_COLOR,
            'action': save_route
        })
        
        def toggle_distance():
            self.show_distance = not self.show_distance
        self.buttons.append({
            'rect': pygame.Rect(CONTROL_START_X + 105, button_y + 45, 80, 35),
            'text': '[Dist]',
            'color': (100, 100, 150),
            'action': toggle_distance
        })
    
    def draw_info_panel(self):
        panel_rect = pygame.Rect(INFO_START_X, 0, INFO_PANEL_WIDTH, HEIGHT)
        pygame.draw.rect(self.screen, CARD_BG, panel_rect)
        pygame.draw.line(self.screen, PRIMARY_COLOR, 
                        (INFO_START_X, 0), (INFO_START_X, HEIGHT), 3)
        
        title = self.title_font.render("PACIENTE SELECIONADO", True, PRIMARY_COLOR)
        self.screen.blit(title, (INFO_START_X + 15, 15))
        
        if self.selected_patient:
            p = self.selected_patient
            
            card_y = 60
            pygame.draw.rect(self.screen, DARK_GRAY,
                           (INFO_START_X + 10, card_y, INFO_PANEL_WIDTH - 20, 220),
                           border_radius=10)
            
            name_text = self.subtitle_font.render(p.name, True, WHITE)
            self.screen.blit(name_text, (INFO_START_X + 20, card_y + 10))
            
            pygame.draw.line(self.screen, MEDIUM_GRAY,
                           (INFO_START_X + 20, card_y + 40),
                           (INFO_START_X + INFO_PANEL_WIDTH - 20, card_y + 40), 1)
            
            urgency_text = self.normal_font.render(f"{p.urgency_level.icon} Urgencia: {p.urgency_level.display_name}",
                                                   True, p.get_color())
            self.screen.blit(urgency_text, (INFO_START_X + 20, card_y + 55))
            
            cat_text = self.normal_font.render(f"{p.medical_category.icon} {p.medical_category.display_name}",
                                              True, LIGHT_GRAY)
            self.screen.blit(cat_text, (INFO_START_X + 20, card_y + 80))
            
            priority_color = SUCCESS_COLOR if p.priority_score > 70 else WARNING_COLOR
            priority_text = self.normal_font.render(f"Prioridade: {p.priority_score:.0f}/100",
                                                    True, priority_color)
            self.screen.blit(priority_text, (INFO_START_X + 20, card_y + 105))
            
            # Barra de prioridade
            bar_rect = pygame.Rect(INFO_START_X + 20, card_y + 125, INFO_PANEL_WIDTH - 40, 8)
            pygame.draw.rect(self.screen, MEDIUM_GRAY, bar_rect, border_radius=4)
            fill_rect = pygame.Rect(INFO_START_X + 20, card_y + 125, 
                                   (INFO_PANEL_WIDTH - 40) * p.priority_score / 100, 8)
            pygame.draw.rect(self.screen, priority_color, fill_rect, border_radius=4)
            
            time_text = self.normal_font.render(f"Tempo estimado: {p.estimated_service_time} min",
                                               True, LIGHT_GRAY)
            self.screen.blit(time_text, (INFO_START_X + 20, card_y + 145))
            
            addr_lines = [p.address[i:i+20] for i in range(0, len(p.address), 20)]
            for i, line in enumerate(addr_lines[:2]):
                addr_text = self.small_font.render(f"End: {line}", True, LIGHT_GRAY)
                self.screen.blit(addr_text, (INFO_START_X + 20, card_y + 175 + i * 18))
            
            phone_text = self.small_font.render(f"Tel: {p.phone}", True, LIGHT_GRAY)
            self.screen.blit(phone_text, (INFO_START_X + 20, card_y + 215))
            
        else:
            no_select = self.normal_font.render("Clique em um paciente", True, MEDIUM_GRAY)
            self.screen.blit(no_select, (INFO_START_X + 20, 100))
            no_select2 = self.small_font.render("para ver detalhes", True, MEDIUM_GRAY)
            self.screen.blit(no_select2, (INFO_START_X + 20, 125))
    
    def draw_control_panel(self):
        panel_rect = pygame.Rect(CONTROL_START_X, 0, CONTROL_PANEL_WIDTH, HEIGHT)
        pygame.draw.rect(self.screen, CARD_BG, panel_rect)
        pygame.draw.line(self.screen, PRIMARY_COLOR,
                        (CONTROL_START_X, 0), (CONTROL_START_X, HEIGHT), 2)
        
        title = self.title_font.render("CONTROLES", True, PRIMARY_COLOR)
        self.screen.blit(title, (CONTROL_START_X + 15, 15))
        
        for button in self.buttons:
            if button['text'] in ['[Pause]', '[Start]']:
                button['text'] = '[Pause]' if not self.paused else '[Start]'
            
            mouse_pos = pygame.mouse.get_pos()
            is_hover = button['rect'].collidepoint(mouse_pos)
            color = button['color']
            if is_hover:
                color = tuple(min(255, c + 30) for c in color)
            
            pygame.draw.rect(self.screen, color, button['rect'], border_radius=6)
            text = self.normal_font.render(button['text'], True, WHITE)
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)
            button['is_hover'] = is_hover
        
        # Estatísticas do GA
        stats_y = 120
        stats_title = self.subtitle_font.render("EVOLUCAO", True, LIGHT_GRAY)
        self.screen.blit(stats_title, (CONTROL_START_X + 15, stats_y))
        
        # Distância total formatada
        distance_text = f"Distancia: {self.ga.best_distance:.0f} px"
        if self.ga.best_distance > 0:
            # Converter para km aproximado (assumindo 1 pixel = 0.1 km)
            distance_km = self.ga.best_distance * 0.1
            distance_text = f"Distancia: {distance_km:.1f} km"
        
        stats = [
            f"Geracao: {self.ga.generation}",
            f"Fitness: {self.ga.best_fitness:.1f}",
            distance_text,
            f"Populacao: {len(self.ga.population)}",
        ]
        
        for i, stat in enumerate(stats):
            color = SUCCESS_COLOR if "km" in stat and self.show_distance else WHITE
            text = self.small_font.render(stat, True, color)
            self.screen.blit(text, (CONTROL_START_X + 15, stats_y + 30 + i * 22))
        
        # Métricas de prioridade
        priority_y = stats_y + 130
        priority_title = self.subtitle_font.render("PRIORIDADE", True, LIGHT_GRAY)
        self.screen.blit(priority_title, (CONTROL_START_X + 15, priority_y))
        
        priority_stats = [
            f"Emerg. inicio: {self.ga.priority_metrics['emergency_first'] * 100:.0f}%",
            f"Cobertura: {self.ga.priority_metrics['priority_coverage'] * 100:.0f}%"
        ]
        
        for i, stat in enumerate(priority_stats):
            text = self.small_font.render(stat, True, WARNING_COLOR)
            self.screen.blit(text, (CONTROL_START_X + 15, priority_y + 30 + i * 22))
        
        # Legenda
        legend_y = HEIGHT - 160
        legend_title = self.small_font.render("LEGENDA", True, LIGHT_GRAY)
        self.screen.blit(legend_title, (CONTROL_START_X + 15, legend_y))
        
        legend_items = [
            (COLOR_EMERGENCY, "Emergencia"),
            (COLOR_HIGH, "Alta"),
            (COLOR_MEDIUM, "Media"),
            (COLOR_LOW, "Baixa"),
            (COLOR_ROUTINE, "Rotina")
        ]
        
        for i, (color, label) in enumerate(legend_items):
            pygame.draw.circle(self.screen, color, (CONTROL_START_X + 25, legend_y + 25 + i * 20), 6)
            text = self.small_font.render(label, True, LIGHT_GRAY)
            self.screen.blit(text, (CONTROL_START_X + 40, legend_y + 22 + i * 20))
    
    def draw_progress_panel(self):
        panel_rect = pygame.Rect(MAP_WIDTH - 280, 15, 265, 110)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, PRIMARY_COLOR, panel_rect, 2, border_radius=10)
        
        title = self.small_font.render("EVOLUCAO DA ROTA", True, PRIMARY_COLOR)
        self.screen.blit(title, (panel_rect.x + 10, panel_rect.y + 8))
        
        if len(self.ga.fitness_history) > 1:
            # Gráfico de Fitness
            history = self.ga.fitness_history[-50:]
            if history:
                max_fitness = max(history)
                min_fitness = min(history)
                range_fitness = max_fitness - min_fitness or 1
                
                graph_rect = pygame.Rect(panel_rect.x + 10, panel_rect.y + 28, 245, 35)
                
                # Título do gráfico
                fit_label = self.small_font.render("Fitness", True, LIGHT_GRAY)
                self.screen.blit(fit_label, (graph_rect.x, graph_rect.y - 12))
                
                for i in range(1, len(history)):
                    x1 = graph_rect.x + (i-1) * graph_rect.width / (len(history) - 1)
                    y1 = graph_rect.y + graph_rect.height - (history[i-1] - min_fitness) * graph_rect.height / range_fitness
                    x2 = graph_rect.x + i * graph_rect.width / (len(history) - 1)
                    y2 = graph_rect.y + graph_rect.height - (history[i] - min_fitness) * graph_rect.height / range_fitness
                    
                    color = SUCCESS_COLOR if history[i] < history[i-1] else PRIMARY_COLOR
                    pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 2)
            
            # Gráfico de Distância
            if len(self.ga.distance_history) > 1:
                dist_history = self.ga.distance_history[-50:]
                max_dist = max(dist_history)
                min_dist = min(dist_history)
                range_dist = max_dist - min_dist or 1
                
                graph_rect = pygame.Rect(panel_rect.x + 10, panel_rect.y + 70, 245, 30)
                
                # Título do gráfico
                dist_label = self.small_font.render("Distancia", True, LIGHT_GRAY)
                self.screen.blit(dist_label, (graph_rect.x, graph_rect.y - 12))
                
                for i in range(1, len(dist_history)):
                    x1 = graph_rect.x + (i-1) * graph_rect.width / (len(dist_history) - 1)
                    y1 = graph_rect.y + graph_rect.height - (dist_history[i-1] - min_dist) * graph_rect.height / range_dist
                    x2 = graph_rect.x + i * graph_rect.width / (len(dist_history) - 1)
                    y2 = graph_rect.y + graph_rect.height - (dist_history[i] - min_dist) * graph_rect.height / range_dist
                    
                    color = SUCCESS_COLOR if dist_history[i] < dist_history[i-1] else (255, 165, 0)
                    pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 2)
    
    def draw_distance_overlay(self):
        """Desenha um overlay mostrando a distância total da rota"""
        if not self.show_distance or not self.ga.best_solution:
            return
        
        # Calcular distância total
        total_distance = self.ga.best_distance
        distance_km = total_distance * 0.1  # Conversão aproximada
        
        # Criar um card flutuante
        overlay_rect = pygame.Rect(MAP_WIDTH - 250, HEIGHT - 80, 240, 65)
        pygame.draw.rect(self.screen, (0, 0, 0, 200), overlay_rect, border_radius=8)
        pygame.draw.rect(self.screen, SUCCESS_COLOR, overlay_rect, 2, border_radius=8)
        
        # Texto da distância
        dist_title = self.small_font.render("DISTANCIA TOTAL", True, SUCCESS_COLOR)
        self.screen.blit(dist_title, (overlay_rect.x + 10, overlay_rect.y + 8))
        
        dist_value = self.big_font.render(f"{distance_km:.1f} km", True, WHITE)
        self.screen.blit(dist_value, (overlay_rect.x + 10, overlay_rect.y + 28))
        
        dist_pixels = self.small_font.render(f"({total_distance:.0f} pixels)", True, LIGHT_GRAY)
        self.screen.blit(dist_pixels, (overlay_rect.x + 10, overlay_rect.y + 48))
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.ga = PriorityGeneticAlgorithm(self.patients, population_size=150)
                    self.ga.initialize_population()
                    self.paused = False
                elif event.key == pygame.K_d:
                    self.show_distance = not self.show_distance
            
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in self.buttons:
                    if button['rect'].collidepoint(event.pos):
                        button['action']()
                
                if event.pos[0] < MAP_WIDTH:
                    self.select_patient_at_position(event.pos)
    
    def select_patient_at_position(self, pos):
        min_dist = 15
        for patient in self.patients:
            dx = patient.location[0] - pos[0]
            dy = patient.location[1] - pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                self.selected_patient = patient
                min_dist = dist
                self.visualizer.add_particle(patient.location[0], patient.location[1], 
                                            patient.get_color())
    
    def update(self):
        if not self.paused and self.ga.generation < 200:
            evolution_data = self.ga.evolve_generation()
            
            if evolution_data['improved']:
                self.route_evolution.add_snapshot(self.ga.best_solution, 
                                                  self.ga.best_fitness,
                                                  self.ga.best_distance,
                                                  self.ga.generation)
                
                if self.ga.best_solution:
                    for i in range(min(5, len(self.ga.best_solution) - 1)):
                        p1 = self.ga.best_solution[i]
                        p2 = self.ga.best_solution[i + 1]
                        self.visualizer.add_connection_highlight(p1, p2)
        
        self.visualizer.update_animation()
        self.route_evolution.update_improvements()
    
    def draw(self):
        self.screen.fill(BLACK)
        
        # Fundo do mapa
        pygame.draw.rect(self.screen, (15, 15, 25), (0, 0, MAP_WIDTH, HEIGHT))
        
        # Grade
        for x in range(0, MAP_WIDTH, 50):
            pygame.draw.line(self.screen, (30, 30, 40), (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, 50):
            pygame.draw.line(self.screen, (30, 30, 40), (0, y), (MAP_WIDTH, y), 1)
        
        # Rota e pacientes
        if self.ga.best_solution:
            self.visualizer.draw_route(self.ga.best_solution)
        
        for patient in self.patients:
            is_selected = (self.selected_patient == patient)
            self.visualizer.draw_patient(patient, is_selected, True)
        
        # Painéis
        self.draw_info_panel()
        self.draw_control_panel()
        self.draw_progress_panel()
        self.draw_distance_overlay()
        
        # Status
        status_text = "[RUNNING]" if not self.paused else "[PAUSED]"
        status_color = SUCCESS_COLOR if not self.paused else WARNING_COLOR
        status_surf = self.big_font.render(status_text, True, status_color)
        self.screen.blit(status_surf, (15, 15))
        
        # FPS
        fps_text = self.small_font.render(f"FPS: {int(self.clock.get_fps())}", True, LIGHT_GRAY)
        self.screen.blit(fps_text, (15, HEIGHT - 25))
        
        # Instruções
        help_text = self.small_font.render("SPACE: Pause | R: Reset | D: Distancia | ESC: Exit", True, MEDIUM_GRAY)
        self.screen.blit(help_text, (15, HEIGHT - 45))
    
    def run(self):
        while self.running and self.ga.generation < 200:
            self.handle_events()
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Mostrar resultado final
        if self.ga.best_solution:
            print("\n" + "="*50)
            print("RESULTADO FINAL")
            print("="*50)
            print(f"Melhor fitness: {self.ga.best_fitness:.2f}")
            print(f"Distancia total: {self.ga.best_distance:.1f} pixels ({self.ga.best_distance * 0.1:.1f} km)")
            print(f"Total de geracoes: {self.ga.generation}")
            print(f"Pacientes atendidos: {len(self.ga.best_solution)}")
            
            print("\nOrdem de atendimento (prioridade):")
            for i, patient in enumerate(self.ga.best_solution[:10]):
                print(f"  {i+1}. {patient.name} - {patient.urgency_level.display_name} (Prioridade: {patient.priority_score:.0f})")
            
            if len(self.ga.best_solution) > 10:
                print(f"  ... e mais {len(self.ga.best_solution) - 10} pacientes")
        
        # Aguardar
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False
            self.clock.tick(10)
        
        pygame.quit()
        sys.exit()

# ==================== PONTO DE ENTRADA ====================

if __name__ == "__main__":
    print("=" * 60)
    print("SISTEMA DE ROTEAMENTO MEDICO - SAUDE DA MULHER")
    print("Visualizacao em Tempo Real com Metrica de Distancia")
    print("=" * 60)
    print("\nCaracteristicas:")
    print("[OK] Priorizacao automatica de pacientes por nivel de urgencia")
    print("[OK] Visualizacao em tempo real da evolucao da melhor rota")
    print("[OK] Metrica de distancia total em pixels e km")
    print("[OK] Grafico de evolucao da distancia")
    print("[OK] Efeitos visuais para melhorias e emergencias")
    print("[OK] Interface interativa com selecao de pacientes")
    print("\nControles:")
    print("  SPACE - Pausar/Continuar evolucao")
    print("  R - Resetar algoritmo")
    print("  D - Mostrar/Esconder distancia")
    print("  ESC - Sair")
    print("  Clique no mapa - Selecionar paciente")
    print("\nIniciando visualizacao...\n")
    
    system = HealthcareRoutingSystem()
    system.run()
