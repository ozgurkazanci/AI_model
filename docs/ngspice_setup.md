# ngspice Kurulum ve Entegrasyon Kılavuzu

## Windows'ta ngspice Kurulumu

### Yöntem 1: Resmi İndirme (Önerilen)

1. https://sourceforge.net/projects/ngspice/files/ adresine git
2. `ngspice-43_64.zip` (veya en son sürüm) indir
3. `C:\ngspice` klasörüne çıkart
4. `C:\ngspice\bin` dizinini `PATH` ortam değişkenine ekle:
   ```powershell
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\ngspice\bin", [EnvironmentVariableTarget]::User)
   ```
5. Doğrula: `ngspice --version`

### Yöntem 2: Chocolatey ile
```powershell
choco install ngspice
```

### Yöntem 3: conda ile
```bash
conda install -c conda-forge ngspice
```

## Doğrulama Testi
```powershell
cd "C:\Users\ozgur\OneDrive - Akdeniz Üniversitesi\CV_akademik_01192017\Arastirmalarim\AI_model"
$env:PYTHONPATH = "src"
python -m pytest tests/test_ngspice_smoke.py -v
```

## Proje Konfigürasyonu

`configs/model_config.yaml` dosyasında ngspice yolunu ayarlayın:
```yaml
simulator:
  backend: ngspice
  binary_path: "C:/ngspice/bin/ngspice.exe"
  work_dir: "./sim_workdir"
  timeout: 60
```
