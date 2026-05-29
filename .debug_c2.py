import tempfile
from pathlib import Path
from backend.ml.model import save_model, list_models, load_model, ModelMetadata
from sklearn.ensemble import RandomForestClassifier
import numpy as np

with tempfile.TemporaryDirectory() as tmpdir:
    models_dir = Path(tmpdir)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(np.random.rand(50, 5), np.random.randint(0, 2, 50))
    metadata = ModelMetadata(feature_columns=['a'], model_type='rf', params={})
    save_model(model, 'test_c2', metadata, models_dir)

    files = sorted(str(f.name) for f in models_dir.glob('*'))
    print('Files:', files)
    
    matches = [str(f.name) for f in models_dir.glob('*_metadata.joblib')]
    print('Glob *_metadata.joblib:', matches)
    
    try:
        entries = list_models(models_dir)
        for e in entries:
            nm = e['name']
            print(f'Entry: {nm} v{e["version"]} - {e["model_type"]}')
    except Exception as ex:
        print(f'ERROR: {ex}')
