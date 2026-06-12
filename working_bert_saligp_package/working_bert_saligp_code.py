
import os
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


BASE_DIR = Path.cwd()
INPUT_DIR = BASE_DIR / "input_files"
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def create_pdf(filename, size_mb, similar=False):
    path = INPUT_DIR / filename

    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4

    common_text = (
        "Once upon a time, a curious cloud travelled across mountains, rivers, schools, "
        "and villages. It collected stories from people and shared knowledge with everyone. "
        "This paragraph is repeated to create duplicate content for BERT duplicate detection. "
    )

    changed_text = (
        "The cloud also learned about cloud security, AES encryption, Bloom filters, "
        "data deduplication, and artificial intelligence. "
    )

    for page in range(1, 20):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, f"Story Dataset Page {page}")

        c.setFont("Helvetica", 10)
        y = height - 90

        for line in range(35):
            if similar and page % 5 == 0:
                text = changed_text + f" Modified page {page}, line {line}."
            else:
                text = common_text + f" Page {page}, line {line}."

            c.drawString(50, y, text[:110])
            y -= 15

        c.showPage()

    c.save()

    target = size_mb * 1024 * 1024
    current = path.stat().st_size

    if current < target:
        with open(path, "ab") as f:
            remaining = target - current
            chunk = b" DUPLICATE STORY CLOUD SECURITY AES BERT SALIGP BLOOM FILTER TEXT DATA\n" * 1024

            while remaining > 0:
                data = chunk[:min(len(chunk), remaining)]
                f.write(data)
                remaining -= len(data)

    return path


def extract_text(pdf_path):
    text = ""
    reader = PdfReader(str(pdf_path))

    for page in reader.pages[:10]:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return text.strip()


def aes_encrypt_decrypt_time(file_path):
    key = b"1234567890123456"

    with open(file_path, "rb") as f:
        data = f.read()

    iv = os.urandom(16)

    start = time.perf_counter()
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv))
    encryptor = cipher.encryptor()
    encrypted = iv + encryptor.update(data) + encryptor.finalize()
    enc_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    cipher = Cipher(algorithms.AES(key), modes.CFB(encrypted[:16]))
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(encrypted[16:]) + decryptor.finalize()
    dec_time = (time.perf_counter() - start) * 1000

    return enc_time, dec_time


print("Creating input files...")

files = [
    create_pdf("storybook_5MB.pdf", 5),
    create_pdf("storybook_10MB.pdf", 10),
    create_pdf("storybook_15MB.pdf", 15),
    create_pdf("storybook_20MB.pdf", 20),
    create_pdf("storybook_25MB.pdf", 25),
    create_pdf("storybook_10MB_similar.pdf", 10, similar=True),
]

for f in files:
    print(f.name, round(f.stat().st_size / (1024 * 1024), 2), "MB")


print("\nRunning AES encryption/decryption timing...")

aes_results = []

for f in files[:5]:
    enc_time, dec_time = aes_encrypt_decrypt_time(f)

    aes_results.append({
        "File": f.name,
        "Size (MB)": round(f.stat().st_size / (1024 * 1024), 2),
        "Encryption Time (ms)": round(enc_time, 4),
        "Decryption Time (ms)": round(dec_time, 4)
    })

df_aes = pd.DataFrame(aes_results)
print(df_aes)
df_aes.to_csv(OUTPUT_DIR / "aes_timing_results.csv", index=False)


print("\nLoading BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


print("\nRunning BERT similarity detection...")

text1 = extract_text(INPUT_DIR / "storybook_10MB.pdf")
text2 = extract_text(INPUT_DIR / "storybook_10MB_similar.pdf")

emb1 = model.encode([text1])
emb2 = model.encode([text2])

similarity = cosine_similarity(emb1, emb2)[0][0] * 100

bert_results = pd.DataFrame([
    {
        "File 1": "storybook_10MB.pdf",
        "File 2": "storybook_10MB_similar.pdf",
        "BERT Similarity (%)": round(similarity, 2),
        "Result": "Duplicate" if similarity >= 90 else "Unique"
    }
])

print(bert_results)
bert_results.to_csv(OUTPUT_DIR / "bert_similarity_results.csv", index=False)


print("\nPaper table values used for comparison...")

table4 = pd.DataFrame({
    "Size (MB)": [5, 10, 15, 20, 25],
    "Paper Encryption Time (ms)": [3.92, 5.83, 11.90, 19.28, 34.92],
    "Paper Decryption Time (ms)": [3.87, 5.02, 10.60, 17.29, 32.82]
})

table5 = pd.DataFrame({
    "Files": [100, 200, 300, 400, 500],
    "BERT": [95.5, 97.6, 96.8, 89.6, 97.4],
    "YOLOv8": [92.14, 93.14, 89.15, 85.91, 86.42],
    "AES": [86.89, 70.86, 86.32, 78.44, 85.08],
    "SALIGP with Bloom Filter": [79.16, 62.53, 64.15, 74.23, 63.71]
})

table6 = pd.DataFrame({
    "Files": [100, 200, 300, 400, 500],
    "BERT": [14.3, 17.6, 11.6, 13.6, 17.6],
    "YOLOv8": [16.5, 19.4, 22.5, 25.9, 19.4],
    "AES": [18.9, 27.8, 25.6, 27.4, 22.8],
    "SALIGP with Bloom Filter": [20.6, 22.5, 16.5, 18.3, 25.3]
})

print(table4)
print(table5)
print(table6)

table4.to_csv(OUTPUT_DIR / "paper_table4.csv", index=False)
table5.to_csv(OUTPUT_DIR / "paper_table5.csv", index=False)
table6.to_csv(OUTPUT_DIR / "paper_table6.csv", index=False)


plt.figure(figsize=(8, 5))
plt.plot(table4["Size (MB)"], table4["Paper Encryption Time (ms)"], marker="o", label="Encryption Time")
plt.plot(table4["Size (MB)"], table4["Paper Decryption Time (ms)"], marker="s", label="Decryption Time")
plt.xlabel("Data Size (MB)")
plt.ylabel("Time (ms)")
plt.title("Paper Table 4: Encryption and Decryption Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "graph_table4_encryption_decryption.png", dpi=200)
plt.show()


plt.figure(figsize=(8, 5))
for col in ["BERT", "YOLOv8", "AES", "SALIGP with Bloom Filter"]:
    plt.plot(table5["Files"], table5[col], marker="o", label=col)

plt.xlabel("Number of Files")
plt.ylabel("Detection Ratio (%)")
plt.title("Paper Table 5: Duplication Detection Ratio")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "graph_table5_detection_ratio.png", dpi=200)
plt.show()


plt.figure(figsize=(8, 5))
for col in ["BERT", "YOLOv8", "AES", "SALIGP with Bloom Filter"]:
    plt.plot(table6["Files"], table6[col], marker="o", label=col)

plt.xlabel("Number of Files")
plt.ylabel("Detection Time (Seconds)")
plt.title("Paper Table 6: Duplicate Detection Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "graph_table6_detection_time.png", dpi=200)
plt.show()


plt.figure(figsize=(6, 4))
plt.bar(bert_results["File 2"], bert_results["BERT Similarity (%)"])
plt.ylabel("Similarity (%)")
plt.title("Measured BERT Similarity")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "measured_bert_similarity.png", dpi=200)
plt.show()


print("\nCompleted successfully.")
print("Input files are inside:", INPUT_DIR)
print("Output CSV and graph files are inside:", OUTPUT_DIR)
