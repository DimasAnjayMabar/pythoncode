import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Dict, Tuple
import time
from deprecated import deprecated

# Definisi kelas untuk Peti Kemas
class PetiKemas:
    def __init__(self, id: int, bobot: float, jenis: str, kota_tujuan: str, jarak_tujuan: int):
        self.id = id
        self.bobot = bobot  # dalam ton
        self.jenis = jenis  # "biasa" atau "khusus"
        self.kota_tujuan = kota_tujuan
        self.jarak_tujuan = jarak_tujuan  # jarak dalam km

# Definisi kelas untuk Kapal
class Kapal:
    def __init__(self, id: int, nama: str, tonase_maksimal: float, kapasitas_khusus: int):
        self.id = id
        self.nama = nama
        self.tonase_maksimal = tonase_maksimal
        self.kapasitas_khusus = kapasitas_khusus  # jumlah peti kemas khusus yang bisa diamankan

# Kelas utama untuk Algoritma Genetika
class GeneticAlgorithmContainerOptimization:
    def __init__(self, 
                 peti_kemas_list: List[PetiKemas], # list peti kemas
                 kapal: Kapal, # 
                 populasi_size: int,
                 max_generations: int,
                 crossover_rate: float,
                 mutation_rate: float,
                 early_stopping: int,
                 tarif_biasa: int = 2000000,  # 2 juta per ton
                 tarif_khusus: int = 10000000,  # 10 juta per ton
                 penalti_tonase: int = 20000000,  # 20 juta per ton kelebihan
                 penalti_urutan: int = 5000000,  # 5 juta per peti salah urutan
                 penalti_khusus: int = 15000000  # 15 juta per peti khusus salah lokasi
                ):
        self.peti_kemas_list = peti_kemas_list
        self.kapal = kapal
        self.populasi_size = populasi_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.early_stopping = early_stopping
        
        self.tarif_biasa = tarif_biasa
        self.tarif_khusus = tarif_khusus
        self.penalti_tonase = penalti_tonase
        self.penalti_urutan = penalti_urutan
        self.penalti_khusus = penalti_khusus
        
        self.peti_ids = [peti.id for peti in peti_kemas_list]
        self.num_peti = len(peti_kemas_list)
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def create_initial_population(self) -> List[List[int]]:
        """Membuat populasi awal dengan permutasi acak dari urutan peti kemas."""
        population = [] # Init array
        for _ in range(self.populasi_size):
            # Membuat chromosome random
            chromosome = self.peti_ids.copy()
            random.shuffle(chromosome)
            population.append(chromosome)
        return population
    
    def calculate_fitness(self, chromosome: List[int]) -> float:
        """Menghitung nilai fitness dari kromosom."""
        total_revenue = 0
        total_penalty = 0
        
        # Menyimpan informasi peti kemas dalam dictionary untuk akses cepat
        peti_dict = {peti.id: peti for peti in self.peti_kemas_list}
        
        # Menghitung total tonase
        total_tonase = sum(peti_dict[id_peti].bobot for id_peti in chromosome)
        
        # Penalti kelebihan tonase
        if total_tonase > self.kapal.tonase_maksimal:
            total_penalty += (total_tonase - self.kapal.tonase_maksimal) * self.penalti_tonase
        
        # Memeriksa urutan peti kemas berdasarkan kota tujuan
        jarak_tujuan = [peti_dict[id_peti].jarak_tujuan for id_peti in chromosome]
        urutan_salah = 0
        for i in range(len(chromosome) - 1):
            for j in range(i + 1, len(chromosome)):
                # Jika peti di bawah (i) memiliki tujuan lebih jauh dari peti di atas (j)
                if jarak_tujuan[i] > jarak_tujuan[j]:
                    urutan_salah += 1
        
        total_penalty += urutan_salah * self.penalti_urutan
        
        # Memeriksa penempatan peti kemas khusus
        peti_khusus_count = 0
        for i, id_peti in enumerate(chromosome):
            peti = peti_dict[id_peti]
            
            # Menambahkan pendapatan
            if peti.jenis == "biasa":
                total_revenue += peti.bobot * self.tarif_biasa
            else:  # Khusus
                total_revenue += peti.bobot * self.tarif_khusus
                peti_khusus_count += 1
                
                # Penalti jika peti khusus melebihi kapasitas lokasi aman
                if peti_khusus_count > self.kapal.kapasitas_khusus:
                    total_penalty += self.penalti_khusus
        
        # Fitness adalah pendapatan dikurangi penalti
        fitness = total_revenue - total_penalty
        return max(fitness, 1) 
    
    def tournament_selection(self, population: List[List[int]], fitness_values: List[float], tournament_size: int = 3) -> List[int]:
        """Memilih kromosom melalui tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_index].copy()
    
    def partially_mapped_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Partially Mapped Crossover (PMX) untuk dua parent."""
        size = len(parent1)
        # Pilih dua titik crossover
        point1, point2 = sorted(random.sample(range(size), 2))
        
        # Inisialisasi offspring dengan nilai -1 (placeholder)
        offspring1 = [-1] * size
        offspring2 = [-1] * size
        
        # Salin bagian tengah dari parent ke offspring
        offspring1[point1:point2] = parent1[point1:point2]
        offspring2[point1:point2] = parent2[point1:point2]
        
        # Membuat mapping untuk bagian tengah
        mapping1 = {parent2[i]: parent1[i] for i in range(point1, point2)}
        mapping2 = {parent1[i]: parent2[i] for i in range(point1, point2)}
        
        # Isi bagian lain dari offspring
        for i in range(size):
            if i < point1 or i >= point2:
                # Untuk offspring1
                candidate1 = parent2[i]
                while candidate1 in offspring1:
                    candidate1 = mapping2.get(candidate1, candidate1)
                offspring1[i] = candidate1
                
                # Untuk offspring2
                candidate2 = parent1[i]
                while candidate2 in offspring2:
                    candidate2 = mapping1.get(candidate2, candidate2)
                offspring2[i] = candidate2
        
        return offspring1, offspring2
    
    def swap_mutation(self, chromosome: List[int]) -> List[int]:
        """Mutasi dengan menukar dua gen secara acak."""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome
    
    def elitism(self, population: List[List[int]], fitness_values: List[float], elite_size: int) -> List[List[int]]:
        """Mempertahankan elite_size kromosom terbaik."""
        # Mengurutkan indeks berdasarkan nilai fitness
        elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)[:elite_size]
        return [population[i].copy() for i in elite_indices]
    
    def evolve(self):
        """Run genetic algorithm"""
        # Initialize population
        population = self.create_initial_population()
        
        # Variable untuk tracking best solution
        best_fitness = 0
        best_chromosome = None
        best_generation = -1
        best_population_index = -1

        generations_no_improvement = 0  # Early stopping counter

        for generation in range(self.max_generations):
            # Menghitung fitness setiap chromosome
            fitness_values = [self.calculate_fitness(chromosome) for chromosome in population]

            # Mencari chromosome terbaik di setiap generasi
            best_idx = fitness_values.index(max(fitness_values))
            current_best_fitness = fitness_values[best_idx]
            current_best_chromosome = population[best_idx]

            # Menyimpan fitness histori
            self.best_fitness_history.append(current_best_fitness)
            self.avg_fitness_history.append(sum(fitness_values) / len(fitness_values))
            
            # Top 5 index
            top_5_index = sorted(range(len(fitness_values)), key=lambda i : fitness_values[i], reverse=True)[:5]

            # Print generation information
            print(f"\nKapal : {self.kapal.nama} — Generation {generation + 1}/{self.max_generations}")
            print(f"Best Fitness : {current_best_fitness:,}")
            print("Top 5 population : ")
            for i, index in enumerate(top_5_index, 1):
                print(f"{i}. Index : {index}, Fitness : {fitness_values[index] : ,}")
            print(f"Best Chromosome: {current_best_chromosome} (Index: {best_idx})")

            # Track solution terbaik
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = current_best_chromosome
                best_generation = generation + 1  # Generasi dimulai dari 1 (untuk mempermudah membaca)
                best_population_index = best_idx  # Index populasi
                generations_no_improvement = 0  # Reset improvement counter
            else:
                generations_no_improvement += 1
                if generations_no_improvement >= self.early_stopping:
                    print(f"\nEarly stopping at generation {generation + 1}")
                    break

            # Elitism untuk memastikan tidak ada fitness bagus terlewat dari generasi sebelumnya
            elite_size = int(self.populasi_size * 0.1)
            elites = self.elitism(population, fitness_values, elite_size)

            # Selection + Crossover + Mutation
            offspring = []
            while len(offspring) < self.populasi_size - elite_size:
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)

                if random.random() < self.crossover_rate:
                    child1, child2 = self.partially_mapped_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                offspring.append(self.swap_mutation(child1))
                if len(offspring) < self.populasi_size - elite_size:
                    offspring.append(self.swap_mutation(child2))

            # Create new population
            population = elites + offspring

            # Memastikan size populasi tetap konstan
            if len(population) > self.populasi_size:
                population = population[:self.populasi_size]

        # Final fitness calculation jika tidak ada early stopping
        if generations_no_improvement < self.early_stopping:
            fitness_values = [self.calculate_fitness(chromosome) for chromosome in population]
            best_idx = fitness_values.index(max(fitness_values))
            if fitness_values[best_idx] > best_fitness:
                best_fitness = fitness_values[best_idx]
                best_chromosome = population[best_idx]
                best_generation = self.max_generations  # Jika solusi hadir di generasi terakhir
                best_population_index = best_idx

        # Print final best solution
        print(f"\n=== Best Solution Found ===")
        print(f"Best Fitness: {best_fitness:,}")
        print(f"Best Chromosome: {best_chromosome}")
        print(f"Found at Generation: {best_generation}")
        print(f"Population Index: {best_population_index}")

        return best_chromosome, best_fitness, best_generation, best_population_index

    def decode_solution(self, chromosome: List[int]):
        """Menampilkan solusi dan menjelaskan setiap penalti yang terjadi."""
        peti_dict = {peti.id: peti for peti in self.peti_kemas_list}
        
        print("\n===== SOLUSI OPTIMAL PENEMPATAN PETI KEMAS =====")
        print(f"Kapal: {self.kapal.nama}")
        print(f"Tonase Maksimal: {self.kapal.tonase_maksimal} ton")
        print(f"Kapasitas Peti Khusus: {self.kapal.kapasitas_khusus}")
        print("\nUrutan Penempatan Peti Kemas (dari bawah ke atas):")
        
        total_tonase = 0
        total_revenue = 0
        total_penalty = 0
        peti_khusus_count = 0

        # Tracking detail penalti
        penalty_log = []

        for i, id_peti in enumerate(chromosome):
            peti = peti_dict[id_peti]
            total_tonase += peti.bobot
            
            jenis_str = "BIASA"
            if peti.jenis == "khusus":
                jenis_str = "KHUSUS"
                peti_khusus_count += 1
                revenue = peti.bobot * self.tarif_khusus
            else:
                revenue = peti.bobot * self.tarif_biasa
            
            total_revenue += revenue
            
            status = ""
            if peti.jenis == "khusus" and peti_khusus_count > self.kapal.kapasitas_khusus:
                status = " [TIDAK AMAN]"
                total_penalty += self.penalti_khusus
                penalty_log.append(f"Peti #{peti.id} melebihi kapasitas khusus → penalti Rp {self.penalti_khusus:,}")
            
            print(f"{i+1}. Peti #{peti.id} - {peti.bobot} ton - {jenis_str} - Tujuan: {peti.kota_tujuan} ({peti.jarak_tujuan} km) - Revenue: Rp {revenue:,}{status}")
        
        print(f"\nTotal Tonase: {total_tonase} ton", end="")
        if total_tonase > self.kapal.tonase_maksimal:
            overload = total_tonase - self.kapal.tonase_maksimal
            print(f" [KELEBIHAN {overload} ton]")
            overload_penalty = overload * self.penalti_tonase
            total_penalty += overload_penalty
            penalty_log.append(f"Kelebihan tonase {overload} ton → penalti Rp {overload_penalty:,}")
        else:
            print(f" [SISA {self.kapal.tonase_maksimal - total_tonase} ton]")

        # Cek urutan berdasarkan jarak tujuan
        jarak_tujuan = [peti_dict[id_peti].jarak_tujuan for id_peti in chromosome]
        urutan_salah = 0
        for i in range(len(chromosome) - 1):
            for j in range(i + 1, len(chromosome)):
                if jarak_tujuan[i] > jarak_tujuan[j]:
                    urutan_salah += 1
                    penalty_log.append(
                        f"Peti #{chromosome[i]} (jarak {jarak_tujuan[i]}) diletakkan di bawah Peti #{chromosome[j]} (jarak {jarak_tujuan[j]}) → penalti Rp {self.penalti_urutan:,}"
                    )
        
        if urutan_salah > 0:
            total_penalty += urutan_salah * self.penalti_urutan
            print(f"Peringatan: {urutan_salah} urutan salah (penalti aktif)")
        else:
            print("Urutan peletakan sudah optimal berdasarkan jarak tujuan")
        
        # Summary akhir
        print(f"\nTotal Revenue: Rp {total_revenue:,}")
        print(f"Total Penalti: Rp {total_penalty:,}")
        print(f"Fitness (Revenue - Penalti): Rp {total_revenue - total_penalty:,}")

        # Cetak rincian penalti
        if penalty_log:
            print("\n--- Rincian Penalti ---")
            for detail in penalty_log:
                print(f"- {detail}")
        else:
            print("\nTidak ada penalti dalam solusi ini")

# Contoh penggunaan
def run_example():
    # Create container list
    peti_kemas_list = [
        PetiKemas(1, 5.0, "biasa", "Kota B", 200),
        PetiKemas(2, 3.0, "biasa", "Kota A", 100),
        PetiKemas(3, 4.0, "khusus", "Kota C", 300),
        PetiKemas(4, 6.0, "biasa", "Kota A", 100),
        PetiKemas(5, 2.0, "khusus", "Kota B", 200),
        PetiKemas(6, 7.0, "biasa", "Kota C", 300),
        PetiKemas(7, 4.0, "biasa", "Kota B", 200),
        PetiKemas(8, 5.0, "biasa", "Kota A", 100),
        PetiKemas(9, 3.5, "khusus", "Kota A", 100),
        PetiKemas(10, 6.5, "biasa", "Kota B", 200),
        PetiKemas(11, 4.5, "biasa", "Kota C", 300),
        PetiKemas(12, 5.5, "biasa", "Kota A", 100),
    ]
    
    # Create ship
    kapal1 = Kapal(1, "Kapal Satrio Jaya", 60.0, 2)
    kapal2 = Kapal(2, "Kapal Krisna Jaya", 70.0, 1)
    kapal3 = Kapal(3, "Vincent Gosend", 80.0, 2)
    
    # Initialize Genetic Algorithm
    ga1 = GeneticAlgorithmContainerOptimization(
        peti_kemas_list=peti_kemas_list,
        kapal=kapal1,
        populasi_size=100,
        max_generations=200,
        crossover_rate=0.8,
        mutation_rate=0.15,
        early_stopping=30
    )

    ga2 = GeneticAlgorithmContainerOptimization(
        peti_kemas_list=peti_kemas_list,
        kapal=kapal2,
        populasi_size=100,
        max_generations=200,
        crossover_rate=0.8,
        mutation_rate=0.15,
        early_stopping=30
    )

    ga3 = GeneticAlgorithmContainerOptimization(
        peti_kemas_list=peti_kemas_list,
        kapal=kapal3,
        populasi_size=100,
        max_generations=200,
        crossover_rate=0.8,
        mutation_rate=0.15,
        early_stopping=30
    )
    
    # Menyimpan ga di dalam list
    ga_instances = [ga1]

    # Init list kosong
    best_chromosomes = []
    best_fitnesses = []
    best_generations = []
    best_population_indices = []

    # Run GA
    start_time = time.time()
    for ga in ga_instances:
        best_chromosome, best_fitness, best_generation, best_population_index = ga.evolve()
        ga.decode_solution(best_chromosome)
        best_chromosomes.append(best_chromosome)
        best_fitnesses.append(best_fitness)
        best_generations.append(best_generation)
        best_population_indices.append(best_population_index)
    end_time = time.time()

if __name__ == "__main__":
    run_example()