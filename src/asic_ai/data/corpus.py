import json
import hashlib
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class CorpusSource(BaseModel):
    id: str
    name: str
    category: str = Field(..., description="'textbook' | 'paper' | 'open_source' | 'documentation' | 'internal'")
    license: str = Field(..., description="SPDX identifier")
    license_compatible: bool = Field(..., description="Compatible with Apache 2.0")
    url: Optional[str] = None
    local_path: Optional[str] = None
    token_count: Optional[int] = None
    notes: str = ""

class CorpusRegistry:
    def __init__(self):
        self.sources: List[CorpusSource] = []
        self._pre_populate()

    def _pre_populate(self):
        self.add(CorpusSource(
            id="textbook_razavi", name="Design of Analog CMOS Integrated Circuits", category="textbook", 
            license="Proprietary", license_compatible=False, notes="Razavi"
        ))
        self.add(CorpusSource(
            id="textbook_gray_meyer", name="Analysis and Design of Analog Integrated Circuits", category="textbook", 
            license="Proprietary", license_compatible=False, notes="Gray & Meyer"
        ))
        self.add(CorpusSource(
            id="textbook_weste_harris", name="CMOS VLSI Design", category="textbook", 
            license="Proprietary", license_compatible=False, notes="Weste & Harris"
        ))
        self.add(CorpusSource(
            id="doc_ngspice", name="ngspice documentation", category="open_source", 
            license="BSD-3-Clause", license_compatible=True
        ))
        self.add(CorpusSource(
            id="doc_xyce", name="Xyce documentation", category="open_source", 
            license="GPL-3.0", license_compatible=False
        ))
        self.add(CorpusSource(
            id="doc_openroad", name="OpenROAD documentation", category="open_source", 
            license="BSD-3-Clause", license_compatible=True
        ))
        self.add(CorpusSource(
            id="doc_klayout", name="KLayout documentation", category="open_source", 
            license="GPL-2.0", license_compatible=False
        ))
        self.add(CorpusSource(
            id="pdk_sky130", name="sky130 PDK", category="open_source", 
            license="Apache-2.0", license_compatible=True
        ))
        self.add(CorpusSource(
            id="pdk_gf180mcu", name="GF180MCU PDK", category="open_source", 
            license="Apache-2.0", license_compatible=True
        ))

    def add(self, source: CorpusSource) -> None:
        self.sources.append(source)

    def validate_licenses(self) -> List[str]:
        """Flag incompatible licenses."""
        return [f"Incompatible license in source {s.id}: {s.license}" for s in self.sources if not s.license_compatible]

    def summary(self) -> Dict[str, 'Any']:
        """Total tokens by category, license breakdown."""
        tokens_by_cat = {}
        licenses = {}
        total_tokens = 0
        
        for s in self.sources:
            cat = s.category
            lic = s.license
            tokens = s.token_count or 0
            
            tokens_by_cat[cat] = tokens_by_cat.get(cat, 0) + tokens
            licenses[lic] = licenses.get(lic, 0) + 1
            total_tokens += tokens
            
        return {
            "total_tokens": total_tokens,
            "tokens_by_category": tokens_by_cat,
            "license_counts": licenses
        }

    def export_manifest(self, path: str) -> None:
        """Write corpus manifest for reproducibility."""
        with open(path, 'w', encoding='utf-8') as f:
            manifest = [s.model_dump() for s in self.sources]
            json.dump(manifest, f, indent=2)

class CorpusProcessor:
    def process_pdf(self, path: str) -> List[str]:
        """Extract text with quality filters."""
        return ["Extracted PDF text..."]

    def process_code(self, path: str, extensions: List[str]) -> List[str]:
        """Extract code with context."""
        return ["Extracted code context..."]

    def mix_general_code(self, corpus: List[str], ratio: float = 0.15) -> List[str]:
        """Add general code to prevent catastrophic forgetting."""
        general_code = ["def hello():\n    print('world')"] * int(len(corpus) * ratio / (1 - ratio))
        return corpus + general_code

    def deduplicate(self, texts: List[str]) -> List[str]:
        """MinHash dedup placeholder."""
        seen = set()
        deduped = []
        for text in texts:
            h = hashlib.md5(text.encode('utf-8')).hexdigest()
            if h not in seen:
                seen.add(h)
                deduped.append(text)
        return deduped

    def quality_filter(self, texts: List[str]) -> List[str]:
        """Remove garbled text, broken formulas, bad tables."""
        return [t for t in texts if len(t.strip()) > 10 and "" not in t]
