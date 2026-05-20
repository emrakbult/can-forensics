from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from can_anomaly.config import Paths


LEGACY_MARKDOWN_REPORT = "CAN_Traffic_Anomaly_Detection_Report_TR.md"


def _read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _xml(text: object) -> str:
    return escape(str(text), quote=False)


def _paragraph(text: str, style: str | None = None, bold: bool = False, align: str | None = None) -> str:
    p_pr = []
    if style:
        p_pr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        p_pr.append(f'<w:jc w:val="{align}"/>')
    r_pr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    p_pr_xml = f"<w:pPr>{''.join(p_pr)}</w:pPr>" if p_pr else ""
    return (
        f"<w:p>{p_pr_xml}<w:r>{r_pr}"
        f'<w:t xml:space="preserve">{_xml(text)}</w:t>'
        "</w:r></w:p>"
    )


def _table(headers: list[str], rows: list[list[str]]) -> str:
    def cell(text: str, header: bool = False) -> str:
        fill = '<w:shd w:fill="D9EAF7"/>' if header else ""
        bold = "<w:rPr><w:b/></w:rPr>" if header else ""
        return (
            "<w:tc><w:tcPr>"
            '<w:tcW w:w="2400" w:type="dxa"/>'
            f"{fill}</w:tcPr><w:p><w:r>{bold}"
            f'<w:t xml:space="preserve">{_xml(text)}</w:t>'
            "</w:r></w:p></w:tc>"
        )

    border = (
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        "</w:tblBorders>"
    )
    header_row = "<w:tr>" + "".join(cell(h, header=True) for h in headers) + "</w:tr>"
    body_rows = "".join("<w:tr>" + "".join(cell(c) for c in row) + "</w:tr>" for row in rows)
    return f"<w:tbl><w:tblPr>{border}</w:tblPr>{header_row}{body_rows}</w:tbl>"


def _docx_parts(document_body: str, created_at: str) -> dict[str, str]:
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    {document_body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="900" w:right="900" w:bottom="900" w:left="900" w:header="708" w:footer="708" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="90" w:line="260" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/><w:sz w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="220"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="32"/><w:color w:val="1F4E79"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="180" w:after="80"/><w:keepNext/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="25"/><w:color w:val="1F4E79"/></w:rPr>
  </w:style>
</w:styles>
"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""
    package_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""
    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
</Relationships>
"""
    settings = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:defaultTabStop w:val="720"/>
</w:settings>
"""
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcmitype="http://purl.org/dc/dcmitype/"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>CAN Trafik Anomali Tespiti</dc:title>
  <dc:creator>CAN Monitor Project</dc:creator>
  <cp:lastModifiedBy>CAN Monitor Project</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created_at}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created_at}</dcterms:modified>
</cp:coreProperties>
"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
  xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>CAN Monitor Project</Application>
</Properties>
"""
    return {
        "[Content_Types].xml": content_types,
        "_rels/.rels": package_rels,
        "word/document.xml": document,
        "word/_rels/document.xml.rels": document_rels,
        "word/styles.xml": styles,
        "word/settings.xml": settings,
        "docProps/core.xml": core,
        "docProps/app.xml": app,
    }


def _write_docx(path: Path, body: str) -> None:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with ZipFile(path, "w", ZIP_DEFLATED) as docx:
        for name, content in _docx_parts(body, created_at).items():
            docx.writestr(name, content.encode("utf-8"))


def _build_results_table(metrics: pd.DataFrame, cv_metrics: pd.DataFrame) -> list[list[str]]:
    if metrics.empty:
        return []
    cv_subset = cv_metrics[["model", "cv_macro_f1_mean"]] if not cv_metrics.empty else pd.DataFrame()
    if not cv_subset.empty:
        table = metrics.merge(cv_subset, on="model", how="left")
    else:
        table = metrics.copy()
        table["cv_macro_f1_mean"] = None
    return [
        [
            str(row["model"]),
            _fmt(row.get("accuracy")),
            _fmt(row.get("macro_f1")),
            _fmt(row.get("cv_macro_f1_mean")),
        ]
        for _, row in table.iterrows()
    ]


def _generate_report_body(paths: Paths) -> str:
    metadata = _read_json(paths.metadata_path)
    feature_meta = _read_json(paths.feature_metadata_path)
    unsup = _read_json(paths.unsupervised_metrics_path)
    metrics = pd.read_csv(paths.metrics_path) if paths.metrics_path.exists() else pd.DataFrame()
    cv_metrics = pd.read_csv(paths.cv_metrics_path) if paths.cv_metrics_path.exists() else pd.DataFrame()

    best = metrics.iloc[0].to_dict() if not metrics.empty else {}
    results_rows = _build_results_table(metrics, cv_metrics)

    body: list[str] = []
    body.append(_paragraph("CAN Trafiği Anomali Tespiti Proje Raporu", "Title", align="center"))
    body.append(_paragraph(f"Oluşturulma tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="center"))

    body.append(_paragraph("1. Proje Başlığı", "Heading1"))
    body.append(_paragraph("CAN Bus Trafiğinde Makine Öğrenmesi Tabanlı Anomali ve Saldırı Tespiti"))

    body.append(_paragraph("2. Proje Ekibi", "Heading1"))
    body.append(_paragraph("Ad Soyad: [Doldurunuz]"))
    body.append(_paragraph("Öğrenci Numarası: [Doldurunuz]"))
    body.append(_paragraph("Bölüm / Sınıf: [Doldurunuz]"))
    body.append(_paragraph("Ekipteki Görev(ler): Veri hazırlama, özellik çıkarımı, model eğitimi, arayüz geliştirme ve raporlama"))

    body.append(_paragraph("3. Proje Özeti", "Heading1"))
    body.append(_paragraph(
        "Bu projede araç içi CAN bus mesajlarından olağan dışı trafik ve saldırı pencereleri tespit edilmektedir. "
        "Ham CSV kayıtları temizlenmiş, zaman, CAN ID ve payload baytlarından kayan pencere özellikleri çıkarılmıştır. "
        "Logistic Regression, KNN, Linear SVM, Decision Tree ve Random Forest modelleri aynı veri üzerinde "
        "karşılaştırılmış; en iyi model yeni CAN dosyalarını sınıflandıran çalışan React/FastAPI demosunda kullanılmıştır. "
        "Ayrıca PCA ve K-Means ile denetimsiz öğrenme analizi yapılmıştır."
    ))

    body.append(_paragraph("4. Giriş", "Heading1"))
    body.append(_paragraph(
        "Projenin amacı, yüklenen bir CAN trafiği dosyasında Normal dışı davranışları tespit eden uygulanabilir bir makine öğrenmesi "
        "prototipi geliştirmektir. Sistem her pencereyi Normal, DoS, Fuzzy veya Impersonation olarak sınıflandırır; "
        "Normal dışı pencereleri şüpheli segmentler halinde raporlar. Konu, araç güvenliği problemini gerçek veri, "
        "karşılaştırmalı model eğitimi ve çalışan demo ile birleştirdiği için seçilmiştir. Kapsam; çevrimdışı CSV "
        "analizi, pencere tabanlı özellik çıkarımı, model karşılaştırması ve web tabanlı demo ile sınırlıdır."
    ))

    body.append(_paragraph("5. Kullanılan Yöntem ve Teknolojiler", "Heading1"))
    dataset_line = "Veri seti: Normal, DoS, Fuzzy ve Impersonation sınıflarından oluşan CAN trafik kayıtları."
    if metadata and feature_meta:
        dataset_line = (
            f"Veri seti: {metadata.get('rows')} CAN mesajı, {feature_meta.get('rows')} kayan pencere; "
            f"pencere boyutu/adım: {feature_meta.get('window_size')}/{feature_meta.get('stride')}."
        )
    for line in [
        "Algoritmalar: Logistic Regression, KNN, Linear SVM, Decision Tree, Random Forest, PCA ve K-Means.",
        "Özellikler: zaman aralığı istatistikleri, CAN ID çeşitliliği, baskın ID oranı, payload istatistikleri ve entropi.",
        "Değerlendirme: train/test ayrımı, stratified k-fold cross validation, accuracy, macro F1, precision ve recall.",
        "Teknolojiler: Python, Pandas, NumPy, Scikit-learn, Matplotlib, FastAPI, React ve Vite.",
        dataset_line,
    ]:
        body.append(_paragraph(f"- {line}"))

    body.append(_paragraph("6. Sonuçlar ve Değerlendirme", "Heading1"))
    body.append(_paragraph(
        "Tüm denetimli modeller aynı özellik kümesi ve aynı değerlendirme metrikleriyle karşılaştırılmıştır. "
        "Aşağıdaki tablo test başarısını ve 5 katlı çapraz doğrulama sonucunu özetler."
    ))
    if results_rows:
        body.append(_table(["Model", "Accuracy", "Macro F1", "CV Macro F1"], results_rows))
    if best:
        body.append(_paragraph(
            f"En iyi sonuç {best.get('model')} modeliyle elde edilmiştir. Test macro F1 skoru "
            f"{_fmt(best.get('macro_f1'))} olduğundan uygulamadaki varsayılan dedektör bu modeldir."
        ))
    if unsup:
        body.append(_paragraph(
            f"Denetimsiz analizde {unsup.get('method')} kullanılmıştır. Adjusted Rand Index "
            f"{_fmt(unsup.get('adjusted_rand_index'))}, NMI {_fmt(unsup.get('normalized_mutual_info'))} ve "
            f"silhouette skoru {_fmt(unsup.get('silhouette_score_sample'))} olarak ölçülmüştür."
        ))
    body.append(_paragraph(
        "Confusion matrix ve PCA/K-Means görselleri React arayüzünde ve outputs/experiments klasöründe üretilmektedir. "
        "Bağımsız test dosyalarında normal trafik temiz, DoS/Fuzzy/Impersonation dosyaları ise doğru saldırı tipiyle "
        "şüpheli olarak işaretlenmiştir. Sınırlılık olarak sistem canlı araç akışını değil, CSV dosyalarını analiz eder."
    ))

    body.append(_paragraph("7. Yenilikçilik / Özgünlük Açıklaması", "Heading1"))
    body.append(_paragraph(
        "Proje yalnızca hazır bir sınıflandırıcı çalıştırmak yerine CAN mesajlarından zaman, kimlik çeşitliliği, payload "
        "istatistikleri ve entropi tabanlı pencere özellikleri çıkarmaktadır. Aynı problem üzerinde ders notlarındaki "
        "klasik yöntemler ve denetimsiz PCA/K-Means yaklaşımı birlikte denenmiştir. React arayüzü de modeli "
        "dosya yükleme senaryosunda kullanılabilir bir anomali tespit prototipine dönüştürmektedir."
    ))

    body.append(_paragraph("8. Gelecek Çalışmalar", "Heading1"))
    for line in [
        "LSTM veya GRU gibi ek sıralı derin öğrenme modelleri denenebilir.",
        "GridSearchCV / RandomizedSearchCV ile daha kapsamlı hiperparametre optimizasyonu yapılabilir.",
        "Gerçek araçtan canlı CAN akışı üzerinde çevrim içi tahmin demosu geliştirilebilir.",
        "Dosya bazlı veya araç bazlı farklı train/test ayrımlarıyla genelleme başarısı ayrıca ölçülebilir.",
    ]:
        body.append(_paragraph(f"- {line}"))

    body.append(_paragraph("9. Kaynakça", "Heading1"))
    for line in [
        "Ders notları: sınıflandırma, KNN/SVM, k-fold cross validation, PCA ve K-Means.",
        "Scikit-learn dokümantasyonu.",
        "FastAPI, React ve Vite dokümantasyonları.",
    ]:
        body.append(_paragraph(f"- {line}"))

    return "\n".join(body)


def report_preview(paths: Paths) -> str:
    if not paths.report_path.exists():
        return ""
    return (
        "# Proje Raporu\n\n"
        f"Teslim raporu DOCX formatında oluşturuldu: `{paths.report_path.name}`.\n\n"
        "Rapor, classnotes içindeki örnek rapor başlık sırasını takip eder ve sonuç tablolarını içerir."
    )


def generate_report(paths: Paths) -> str:
    """Generate a Turkish DOCX project report aligned with the course template."""

    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    body = _generate_report_body(paths)
    _write_docx(paths.report_path, body)

    legacy_path = paths.reports_dir / LEGACY_MARKDOWN_REPORT
    if legacy_path.exists():
        legacy_path.unlink()

    return report_preview(paths)

