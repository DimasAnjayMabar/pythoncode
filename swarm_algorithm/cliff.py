import random
import numpy as np

# ===================== Struktur Data Awal =====================
# Contoh data perawat (ID 1 - 10)
perawat = [{"id": i, "nama": f"Perawat {i}", "sertifikat": ["ICU"] if i % 3 == 0 else []} for i in range(1, 11)]

# 8 Tipe Bangsal (asumsi ID 0-7)
bangsal = [
    {"id": 1, "nama": "ICU", "maksimal": 4, "perawat": []},
    {"id": 2, "nama": "Penyakit Menular", "maksimal": 4, "perawat": []},
    {"id": 3, "nama": "Penyakit Tidak Menular", "maksimal": 2, "perawat": []},
    {"id": 4, "nama": "Ibu Melahirkan", "maksimal": 4, "perawat": []},
    {"id": 5, "nama": "Bayi dan Premature", "maksimal": 8, "perawat": []},
    {"id": 6, "nama": "Klinik Umum", "maksimal": 2, "perawat": []},
    {"id": 7, "nama": "Klinik Gigi", "maksimal": 2, "perawat": []},
    {"id": 8, "nama": "IGD", "maksimal": 8, "perawat": []},
]

num_days = 30     # 1 bulan
num_shifts = 3    # Pagi, Sore, Malam
num_bangsal = len(bangsal)

# ===================== Parameter PSO =====================
num_particles = 10
num_iterations = 100
w, c1, c2 = 0.5, 1.5, 1.5

# ===================== Fungsi Generate Partikel =====================
def generate_particle():
    # Random ID perawat (1 - 10) untuk setiap shift dan bangsal per hari
    return np.random.randint(1, 11, (num_days, num_shifts, num_bangsal))

# ===================== Fungsi Fitness =====================
def fitness_function(particle, bangsal_data, perawat_data):
    penalty = 0
    particle = particle.reshape((num_days, num_shifts, num_bangsal))  # Pastikan bentuknya
    kerja_per_perawat_per_minggu = {p['id']: [0]* (num_days // 7 + 1) for p in perawat_data}

    # Cek konflik antar shift dan isi per bangsal
    for day in range(num_days):
        daily_perawat = set()
        for shift in range(num_shifts):
            shift_perawat = set()
            for b in range(num_bangsal):
                perawat_id = int(particle[day][shift][b])
                
                # Cek perawat double shift dalam sehari
                if perawat_id in shift_perawat:
                    penalty += 10  # double job dalam shift
                shift_perawat.add(perawat_id)
                
                # Cek sertifikat ICU
                if bangsal_data[b]['nama'] == 'ICU' and 'ICU' not in perawat_data[perawat_id - 1]['sertifikat']:
                    penalty += 30  # tidak punya sertif
                
                # Simpan kerja harian per perawat
                daily_perawat.add(perawat_id)
                
            # Cek maksimal per bangsal
            if len(shift_perawat) > bangsal_data[b]['maksimal']:
                penalty += 20  # overload bangsal

        # Tambah kerja per minggu
        minggu_ke = day // 7
        for pid in daily_perawat:
            kerja_per_perawat_per_minggu[pid][minggu_ke] += 1

    # Cek libur 2x seminggu
    for pid, mingguan in kerja_per_perawat_per_minggu.items():
        for kerja in mingguan:
            if kerja > 5:
                penalty += (kerja - 5) * 15  # semakin sering dia kerja, penalti makin besar

    return penalty

# ===================== Inisialisasi PSO =====================
particles = [generate_particle() for _ in range(num_particles)]
velocities = [np.zeros_like(p) for p in particles] # GPT dulu
pbest = particles.copy()
pbest_scores = [fitness_function(p, bangsal, perawat) for p in particles]
gbest = pbest[np.argmin(pbest_scores)] # GPT dulu
gbest_score = min(pbest_scores)

# ===================== Proses Iterasi PSO =====================
for iterasi in range(num_iterations):
    print(f"Iterasi {iterasi + 1}")
    for i in range(num_particles):
        # Update velocity
        inertia = w * velocities[i]
        cognitive = c1 * random.random() * (pbest[i] - particles[i])
        social = c2 * random.random() * (gbest - particles[i])
        velocities[i] = inertia + cognitive + social

        # Update posisi (solusi jadwal)
        particles[i] = particles[i] + velocities[i]
        particles[i] = np.clip(particles[i], 1, 10)  # ID perawat tetap di 1-10

        # Hitung fitness baru (❗❗ diperbaiki di sini)
        score = fitness_function(particles[i], bangsal, perawat)

        # Update personal best
        if score < pbest_scores[i]:
            pbest[i] = particles[i].copy()
            pbest_scores[i] = score

    # Update global best
    if min(pbest_scores) < gbest_score:
        gbest = pbest[np.argmin(pbest_scores)].copy()
        gbest_score = min(pbest_scores)

    print(f"Best Fitness Iterasi {iterasi + 1}: {gbest_score}")

# ===================== Output Jadwal Terbaik =====================
print("\n=== Jadwal Perawat Terbaik ===")
for day in range(num_days):
    print(f"\nHari ke-{day + 1}")
    for shift in range(num_shifts):
        shift_name = ["Pagi", "Sore", "Malam"][shift]
        print(f"  Shift {shift_name}:")
        for b in range(num_bangsal):
            perawat_id = int(gbest[day][shift][b])
            print(f"    {bangsal[b]['nama']}: Perawat-{perawat_id}")

print(f"\nFitness Akhir (Total Penalti): {gbest_score}")
