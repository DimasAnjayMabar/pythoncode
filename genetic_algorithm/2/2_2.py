import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Dict, Tuple
import time
from deprecated import deprecated

# ==============================================================================
# Classes
# ==============================================================================

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
                 peti_kemas_list: List[PetiKemas],  # list peti kemas
                 kapal: Kapal,  #
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
        
# ==============================================================================
# Inisialisasi populasi
# ==============================================================================
    def create_initial_population(self) -> List[List[int]]:
        """Membuat populasi awal dengan permutasi acak dari urutan peti kemas."""
        population = []  # Init array
        for _ in range(self.populasi_size):
            # Membuat chromosome random
            chromosome = self.peti_ids.copy()
            random.shuffle(chromosome)
            population.append(chromosome)
        return population

# ==============================================================================
# Fitness function alternative (tidak mengangkut container special jika maksimal tempat container special telah penuh)
# ==============================================================================
    # --- Fungsi Fitness Alternatif ---
    def calculate_fitness_alternative(self, chromosome: List[int]) -> float:
        """
        Menghitung nilai fitness (ALTERNATIF).
        Jika kapasitas khusus penuh, peti khusus berikutnya TIDAK DIANGKUT (diabaikan).
        """
        loaded_revenue = 0
        loaded_tonase = 0
        total_penalty = 0  # Penalti khusus tidak ada di sini, hanya tonase & urutan

        peti_dict = {peti.id: peti for peti in self.peti_kemas_list}

        peti_khusus_loaded_count = 0
        # Kita perlu melacak peti mana saja yang *benar-benar* dimuat untuk cek urutan LIFO
        actually_loaded_ids_in_order = []

        # Iterasi melalui urutan di kromosom, mensimulasikan pemuatan
        for id_peti in chromosome:
            peti = peti_dict[id_peti]

            can_be_loaded = True  # Asumsi awal bisa dimuat

            if peti.jenis == "khusus":
                # Cek apakah kapasitas khusus masih ada
                if peti_khusus_loaded_count >= self.kapal.kapasitas_khusus:
                    # Kapasitas penuh, peti khusus ini TIDAK diangkut
                    can_be_loaded = False
                # else: Masih ada kapasitas, akan dimuat (jika can_be_loaded tetap True)

            # Jika peti bisa dimuat (baik biasa maupun khusus dalam kapasitas)
            if can_be_loaded:
                # Tambahkan ke daftar yang benar-benar dimuat
                actually_loaded_ids_in_order.append(id_peti)

                # Tambahkan bobotnya
                loaded_tonase += peti.bobot

                # Tambahkan revenue-nya
                if peti.jenis == "biasa":
                    loaded_revenue += peti.bobot * self.tarif_biasa
                else:  # Khusus (yang berhasil dimuat)
                    loaded_revenue += peti.bobot * self.tarif_khusus
                    peti_khusus_loaded_count += 1  # Tambah counter HANYA jika berhasil dimuat

        # --- Hitung Penalti berdasarkan peti yang BENAR-BENAR dimuat ---

        # 1. Penalti Kelebihan Tonase (berdasarkan yang dimuat)
        if loaded_tonase > self.kapal.tonase_maksimal:
            total_penalty += (loaded_tonase - self.kapal.tonase_maksimal) * self.penalti_tonase

        # 2. Penalti Urutan (berdasarkan urutan peti yang dimuat)
        urutan_salah = 0
        if len(actually_loaded_ids_in_order) > 1:  # Perlu minimal 2 peti untuk cek urutan
            jarak_tujuan_loaded = [peti_dict[id_peti].jarak_tujuan for id_peti in actually_loaded_ids_in_order]
            for i in range(len(actually_loaded_ids_in_order) - 1):
                for j in range(i + 1, len(actually_loaded_ids_in_order)):
                    # Jika peti yang dimuat lebih awal (i) tujuannya lebih jauh dari peti (j)
                    if jarak_tujuan_loaded[i] > jarak_tujuan_loaded[j]:
                        urutan_salah += 1
            total_penalty += urutan_salah * self.penalti_urutan

        # 3. Penalti Khusus (TIDAK ADA dalam logika ini)

        # Fitness adalah revenue dari yang dimuat dikurangi penalti tonase & urutan
        fitness = loaded_revenue - total_penalty
        return max(fitness, 1)  # Pastikan minimal 1

# ==============================================================================
# Fitness function original (tetap memasukkan container special selama kapal muat, tapi akan diletakkan di tempat  reguler)
# ==============================================================================
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

# ==============================================================================
# Tournament selection untuk memilih parent
# ==============================================================================
    def tournament_selection(self, population: List[List[int]], fitness_values: List[float],
                             tournament_size: int = 3) -> List[int]:
        """Memilih kromosom melalui tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_index].copy()

# ==============================================================================
# Roullete wheel untuk memilih parent
# ==============================================================================
    def roulette_wheel_selection(self, population: List[List[int]], fitness_values: List[float]) -> List[int]:
        """Memilih kromosom (parent) melalui metode Roulette Wheel selection.

        Metode ini memilih individu dengan probabilitas yang proporsional
        terhadap nilai fitnessnya.

        Args:
            population (List[List[int]]): Populasi saat ini.
            fitness_values (List[float]): Daftar nilai fitness untuk setiap
                                         individu di populasi. Fitness harus >= 1.

        Returns:
            List[int]: Kromosom (individu) yang terpilih (sebagai salinan).
                       Mengembalikan list kosong jika populasi kosong.
        """
        if not population:
            print("Error: Populasi kosong saat mencoba Roulette Wheel Selection.")
            return [] # Kembalikan list kosong jika tidak ada populasi

        # Pastikan fitness_values dan population memiliki panjang yang sama
        if len(population) != len(fitness_values):
             raise ValueError("Panjang populasi dan fitness_values tidak cocok.")

        # Hitung total fitness. Diasumsikan semua fitness >= 1 (dari calculate_fitness)
        total_fitness = sum(fitness_values)

        # Penanganan jika total fitness 0 (seharusnya tidak terjadi jika min fitness = 1)
        if total_fitness <= 0:
            # Jika semua fitness 1 (minimum), total fitness akan > 0
            # Ini hanya terjadi jika ada bug atau fitness bisa <= 0
            print("Warning: Total fitness <= 0 dalam Roulette Wheel. Memilih secara acak.")
            return random.choice(population).copy()

        # Hitung probabilitas seleksi relatif untuk setiap individu
        selection_probs = [f / total_fitness for f in fitness_values]

        # Hitung probabilitas kumulatif
        cumulative_probs = [0.0] * len(population)
        current_sum = 0.0
        for i, prob in enumerate(selection_probs):
            current_sum += prob
            cumulative_probs[i] = current_sum

        # Pastikan nilai kumulatif terakhir adalah 1.0 untuk mengatasi
        # potensi error floating point kecil.
        # Ini penting agar random_pick yang bernilai tepat 1.0 masih bisa memilih individu terakhir.
        cumulative_probs[-1] = 1.0

        # "Putar roda" - hasilkan angka acak antara 0.0 dan 1.0
        random_pick = random.random()

        # Cari individu yang sesuai dengan hasil putaran roda
        for i, cumulative_prob in enumerate(cumulative_probs):
            if random_pick <= cumulative_prob:
                # Kembalikan SALINAN kromosom yang terpilih
                return population[i].copy()

        # Fallback - Seharusnya tidak pernah sampai sini jika cumulative_probs[-1] = 1.0
        # Namun, sebagai pengaman, kembalikan individu terakhir.
        print("Warning: Fallback Roulette Wheel tercapai (seharusnya tidak terjadi).")
        return population[-1].copy()

# ==============================================================================
# Partially mapped crossover
# ==============================================================================
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

# ==============================================================================
# Swap mutation
# ==============================================================================
    def swap_mutation(self, chromosome: List[int]) -> List[int]:
        """Mutasi dengan menukar dua gen secara acak."""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome

# ==============================================================================
# Elitism untuk mempertahankan solusi bagus
# ==============================================================================
    def elitism(self, population: List[List[int]], fitness_values: List[float], elite_size: int) -> List[List[int]]:
        """Mempertahankan elite_size kromosom terbaik."""
        # Mengurutkan indeks berdasarkan nilai fitness
        elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)[:elite_size]
        return [population[i].copy() for i in elite_indices]

# ==============================================================================
# Pemanggilan fungsi-fungsi lain yang dijadikan satu menjadi GA
# ==============================================================================
    def evolve(self, selection_method='tournament', fitness_function_type='original'): # Ditambahkan parameter fitness_function_type
        """
        Menjalankan algoritma genetika untuk mencari solusi optimal.

        Args:
            selection_method (str): Metode seleksi ('tournament' atau 'roulette'). Default: 'tournament'.
            fitness_function_type (str): Fungsi fitness yang digunakan ('original' atau 'alternative').
                                         'original': Penalti jika > kapasitas_khusus.
                                         'alternative': Peti khusus > kapasitas_khusus diabaikan.
                                         Default: 'original'.

        Returns:
            Tuple: Berisi (best_chromosome, best_fitness, best_generation,
                   best_population_index, best_fitness_history, avg_fitness_history)
        """
        print(f"\n=== Memulai Evolusi untuk Kapal: {self.kapal.nama} ===")
        print(f"Metode Seleksi: {selection_method.capitalize()}")
        # Menampilkan fungsi fitness yang digunakan
        print(f"Fungsi Fitness: {fitness_function_type.capitalize()}")

        # Inisialisasi populasi
        population = self.create_initial_population()

        # Reset history untuk run ini
        best_fitness_history = []
        avg_fitness_history = []

        # Variabel untuk tracking solusi terbaik keseluruhan
        best_overall_fitness = 0
        best_overall_chromosome = None
        best_generation = -1
        best_population_index = -1

        generations_no_improvement = 0  # Counter untuk early stopping

        # Ukuran elit (misal 10% dari populasi)
        elite_size = max(1, int(self.populasi_size * 0.1))  # Pastikan minimal 1

        # --- Pilih fungsi fitness berdasarkan parameter ---
        if fitness_function_type == 'original':
            calculate_fitness_func = self.calculate_fitness
        elif fitness_function_type == 'alternative':
            # Pastikan fungsi ini sudah ada di dalam kelas!
            if hasattr(self, 'calculate_fitness_alternative'):
                 calculate_fitness_func = self.calculate_fitness_alternative
            else:
                 raise AttributeError("Fungsi 'calculate_fitness_alternative' belum didefinisikan dalam kelas.")
        else:
            raise ValueError(f"Tipe fungsi fitness tidak dikenal: {fitness_function_type}")
        # -------------------------------------------------

        for generation in range(self.max_generations):
            # 1. Evaluasi: Gunakan fungsi fitness yang sudah dipilih
            try:
                # Menggunakan fungsi yang sudah dipilih di atas
                fitness_values = [calculate_fitness_func(chromosome) for chromosome in population]
            except Exception as e:
                print(f"Error calculating fitness in generation {generation + 1} using {fitness_function_type} function: {e}")
                return None, -1, -1, -1, [], []  # Return nilai default jika error

            # Cari solusi terbaik di generasi ini
            # (Penanganan jika fitness_values kosong, misal populasi 0)
            if not fitness_values:
                print(f"Warning: Populasi kosong di generasi {generation + 1}.")
                break # Hentikan jika populasi kosong

            current_best_idx = fitness_values.index(max(fitness_values))
            current_best_fitness = fitness_values[current_best_idx]
            current_best_chromosome = population[current_best_idx]

            # Simpan history fitness
            best_fitness_history.append(current_best_fitness)
            avg_fitness = sum(fitness_values) / len(fitness_values)
            avg_fitness_history.append(avg_fitness)

            # Print info generasi
            print(
                f"\nKapal: {self.kapal.nama} | Generasi {generation + 1}/{self.max_generations} | Seleksi: {selection_method.capitalize()} | Fitness: {fitness_function_type.capitalize()}")
            print(f"Best Fitness Generasi Ini : {current_best_fitness:,.2f} (Index: {current_best_idx})")
            print(f"Avg Fitness Generasi Ini  : {avg_fitness:,.2f}")

            # Track solusi terbaik sepanjang evolusi
            if current_best_fitness > best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_overall_chromosome = current_best_chromosome
                best_generation = generation + 1
                best_population_index = current_best_idx
                generations_no_improvement = 0  # Reset counter
                print(f"*** New Best Overall Fitness Found: {best_overall_fitness:,.2f} ***")
            else:
                generations_no_improvement += 1
                if generations_no_improvement >= self.early_stopping:
                    print(
                        f"\nEarly stopping triggered at generation {generation + 1} (No improvement for {self.early_stopping} generations).")
                    break

            # 2. Elitism: Pertahankan individu terbaik
            elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)[
                            :elite_size]
            elites = [population[i].copy() for i in elite_indices]

            # 3. Seleksi Parent & Reproduksi (Crossover + Mutasi)
            offspring = []
            while len(offspring) < self.populasi_size - elite_size:
                # Pilih parent menggunakan metode yang ditentukan
                try:
                    if selection_method == 'tournament':
                        parent1 = self.tournament_selection(population, fitness_values)
                        parent2 = self.tournament_selection(population, fitness_values)
                    elif selection_method == 'roulette':
                        parent1 = self.roulette_wheel_selection(population, fitness_values)
                        parent2 = self.roulette_wheel_selection(population, fitness_values)
                    else:
                        raise ValueError(f"Metode seleksi tidak dikenal: {selection_method}")
                except Exception as e:
                    print(f"Error during parent selection in generation {generation + 1}: {e}")
                    continue

                # Crossover
                if random.random() < self.crossover_rate:
                    try:
                        child1, child2 = self.partially_mapped_crossover(parent1, parent2)
                    except Exception as e:
                        print(f"Error during crossover in generation {generation + 1}: {e}")
                        child1, child2 = parent1.copy(), parent2.copy()
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutasi
                offspring.append(self.swap_mutation(child1))
                if len(offspring) < self.populasi_size - elite_size:
                    offspring.append(self.swap_mutation(child2))

            # 4. Bentuk Populasi Baru
            population = elites + offspring

            # Pastikan ukuran populasi tetap
            if len(population) != self.populasi_size:
                print(
                    f"Warning: Population size changed to {len(population)} in generation {generation + 1}. Adjusting.")
                population = population[:self.populasi_size]

        # Jika loop selesai tanpa early stopping, cek fitness terakhir sekali lagi
        if generations_no_improvement < self.early_stopping:
            print("\nMencapai Max Generations.")
            # Hitung fitness populasi final menggunakan fungsi yang sama
            final_fitness_values = [calculate_fitness_func(chromosome) for chromosome in population]
            if final_fitness_values: # Cek jika tidak kosong
                final_best_idx = final_fitness_values.index(max(final_fitness_values))
                if final_fitness_values[final_best_idx] > best_overall_fitness:
                    best_overall_fitness = final_fitness_values[final_best_idx]
                    best_overall_chromosome = population[final_best_idx]
                    best_generation = self.max_generations
                    best_population_index = final_best_idx
                    print(f"*** Best solution found in the final generation: {best_overall_fitness:,.2f} ***")

        # Print hasil akhir
        print(f"\n=== Evolusi Selesai untuk Kapal: {self.kapal.nama} ({selection_method.capitalize()} / {fitness_function_type.capitalize()}) ===")
        if best_overall_chromosome:
            print(f"Best Fitness Overall : {best_overall_fitness:,.2f}")
            
            # For alternative fitness, show only loaded containers
            if fitness_function_type == 'alternative':
                peti_dict = {peti.id: peti for peti in self.peti_kemas_list}
                loaded_ids = []
                peti_khusus_loaded = 0
                
                for id_peti in best_overall_chromosome:
                    peti = peti_dict[id_peti]
                    if peti.jenis == "khusus":
                        if peti_khusus_loaded < self.kapal.kapasitas_khusus:
                            loaded_ids.append(id_peti)
                            peti_khusus_loaded += 1
                        else:
                            continue  # Skip this special container
                    else:
                        loaded_ids.append(id_peti)
                
                print(f"Best Chromosome (Loaded Containers) : {loaded_ids}")
            else:
                print(f"Best Chromosome      : {best_overall_chromosome}")
            
            print(f"Ditemukan di Gen     : {best_generation}")
            print(f"Index Populasi       : {best_population_index}")
        else:
            print("Tidak ada solusi valid yang ditemukan.")

        # Return semua informasi penting
        return (best_overall_chromosome, best_overall_fitness, best_generation,
                best_population_index, best_fitness_history, avg_fitness_history)

# ==============================================================================
# Print detail container dalam kapal
# ==============================================================================
    def decode_solution(self, chromosome: List[int], fitness_function_type='original'):
        """Menampilkan solusi dan menjelaskan setiap penalti yang terjadi."""
        peti_dict = {peti.id: peti for peti in self.peti_kemas_list}

        print("\n===== SOLUSI OPTIMAL PENEMPATAN PETI KEMAS =====")
        print(f"Kapal: {self.kapal.nama}")
        print(f"Tonase Maksimal: {self.kapal.tonase_maksimal} ton")
        print(f"Kapasitas Peti Khusus: {self.kapal.kapasitas_khusus}")
        
        if fitness_function_type == 'alternative':
            print("\n[NOTE: Using Alternative Fitness - Special containers beyond capacity are NOT LOADED]")
        print("\nUrutan Penempatan Peti Kemas (dari bawah ke atas):")

        total_tonase = 0
        total_revenue = 0
        total_penalty = 0
        peti_khusus_count = 0

        # Tracking detail penalti
        penalty_log = []
        
        # For alternative fitness, track which containers are actually loaded
        actually_loaded_ids = []
        
        # First pass: determine which containers are actually loaded (for alternative fitness)
        if fitness_function_type == 'alternative':
            peti_khusus_loaded_count = 0
            for id_peti in chromosome:
                peti = peti_dict[id_peti]
                can_be_loaded = True
                
                if peti.jenis == "khusus":
                    if peti_khusus_loaded_count >= self.kapal.kapasitas_khusus:
                        can_be_loaded = False
                
                if can_be_loaded:
                    actually_loaded_ids.append(id_peti)
                    if peti.jenis == "khusus":
                        peti_khusus_loaded_count += 1
        else:
            actually_loaded_ids = chromosome.copy()

        # Second pass: calculate metrics and display only loaded containers
        for i, id_peti in enumerate(actually_loaded_ids):
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

            print(f"{i + 1}. Peti #{peti.id} - {peti.bobot} ton - {jenis_str} - Tujuan: {peti.kota_tujuan} ({peti.jarak_tujuan} km) - Revenue: Rp {revenue:,}{status}")

        # Rest of the method remains the same...
        print(f"\nTotal Tonase: {total_tonase} ton", end="")
        if total_tonase > self.kapal.tonase_maksimal:
            overload = total_tonase - self.kapal.tonase_maksimal
            print(f" [KELEBIHAN {overload} ton]")
            overload_penalty = overload * self.penalti_tonase
            total_penalty += overload_penalty
            penalty_log.append(f"Kelebihan tonase {overload} ton → penalti Rp {overload_penalty:,}")
        else:
            print(f" [SISA {self.kapal.tonase_maksimal - total_tonase} ton]")

        # Cek urutan berdasarkan jarak tujuan (only for loaded containers)
        jarak_tujuan = [peti_dict[id_peti].jarak_tujuan for id_peti in actually_loaded_ids]
        urutan_salah = 0
        for i in range(len(actually_loaded_ids) - 1):
            for j in range(i + 1, len(actually_loaded_ids)):
                if jarak_tujuan[i] > jarak_tujuan[j]:
                    urutan_salah += 1
                    penalty_log.append(
                        f"Peti #{actually_loaded_ids[i]} (jarak {jarak_tujuan[i]}) diletakkan di bawah Peti #{actually_loaded_ids[j]} (jarak {jarak_tujuan[j]}) → penalti Rp {self.penalti_urutan:,}"
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

# ==============================================================================
# Fungsi main
# ==============================================================================
def run_interactive_menu():
    print("=== MENU OPTIMASI GENETIKA ===")
    print("Pilih Metode Seleksi:")
    print("1. Tournament")
    print("2. Roulette")
    selection_choice = input("Masukkan pilihan (1/2): ").strip()

    if selection_choice == '1':
        selection_method = 'tournament'
    elif selection_choice == '2':
        selection_method = 'roulette'
    else:
        print("Pilihan seleksi tidak valid. Default: Tournament")
        selection_method = 'tournament'

    print("\nPilih Fungsi Fitness:")
    print("1. Original")
    print("2. Alternative")
    fitness_choice = input("Masukkan pilihan (1/2): ").strip()

    if fitness_choice == '1':
        fitness_type = 'original'
    elif fitness_choice == '2':
        fitness_type = 'alternative'
    else:
        print("Pilihan fitness tidak valid. Default: Original")
        fitness_type = 'original'

    print(f"\n>> Metode Seleksi: {selection_method.capitalize()}")
    print(f">> Fungsi Fitness: {fitness_type.capitalize()}")

    run_ga(selection_method, fitness_type)

def run_ga(selection_method, fitness_type):
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
    kapal = Kapal(1, "Kapal Interaktif", 60.0, 2)

    ga_instance = GeneticAlgorithmContainerOptimization(
        peti_kemas_list=peti_kemas_list,
        kapal=kapal,
        populasi_size=100,
        max_generations=150,
        crossover_rate=0.85,
        mutation_rate=0.10,
        early_stopping=25
    )

    best_chromo, best_fit, _, _, _, _ = ga_instance.evolve(selection_method, fitness_type)
    if best_chromo:
        print("\n--- Rincian Solusi ---")
        ga_instance.decode_solution(best_chromo, fitness_type)
        print(f"\nFitness ({fitness_type.capitalize()}): Rp {best_fit:,.2f}")
    else:
        print("Tidak ada solusi valid ditemukan.")

# Entry Point
if __name__ == "__main__":
    run_interactive_menu()

