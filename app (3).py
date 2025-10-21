
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import base64
import tempfile
import os
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# ==================== MODEL PATH CONFIGURATION ====================
MODEL_PATH = "model_embedded.txt"  # Google Colab için
# MODEL_PATH = "model_embedded.txt"  # Yerel için
# ================================================================

st.set_page_config(
    page_title="💰 Nakit Akış Sistemi",
    page_icon="💰",
    layout="wide"
)

st.markdown("""
<style>
    .module-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .adaptive-card {
        border-left: 5px solid #28a745;
        padding: 1rem;
        background: #f8f9fa;
        margin: 0.5rem 0;
    }
    # .lstm-card {
    #     border-left: 5px solid #ffc107;
    #     padding: 1rem;
    #     background: #fffbf0;
    #     margin: 0.5rem 0;
    # }
    .warning-card {
        border-left: 5px solid #dc3545;
        padding: 1rem;
        background: #fff0f0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

if 'sistem' not in st.session_state:
    st.session_state.sistem = None
    st.session_state.initialized = False
    st.session_state.auto_refresh = True  # Otomatik yenileme için

class NakitAkisYonetimiTam:
    def __init__(self):
        self.hareketler = pd.DataFrame(columns=['Tarih', 'Aciklama', 'Kategori', 'Tutar', 'Tip', 'Hesap'])
        self.hesaplar = {}
        self.lstm_model = None
        self.model_yuklendi = False
        self.model_hata_mesaji = None
        
        # Otomatik analiz için önbellek
        self.son_adaptif_analiz = None
        self.son_analiz_parametreleri = None
        
        if TENSORFLOW_AVAILABLE:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self._dis_dosyadan_model_yukle()

    def _dis_dosyadan_model_yukle(self):
        """Dış dosyadan Base64 model yükleme"""
        try:
            if not os.path.exists(MODEL_PATH):
                self.model_hata_mesaji = f"⚠️ Model dosyası bulunamadı: {MODEL_PATH}"
                st.info(f"{self.model_hata_mesaji}\n\nLSTM modeli olmadan sadece Adaptif ve Klasik analiz çalışacak.")
                return
            
            try:
                with open(MODEL_PATH, 'r', encoding='utf-8') as f:
                    model_base64_string = f.read().strip()
            except Exception as read_error:
                self.model_hata_mesaji = f"❌ Dosya okuma hatası: {str(read_error)}"
                st.error(self.model_hata_mesaji)
                return
            
            if not model_base64_string or len(model_base64_string) < 100:
                self.model_hata_mesaji = "⚠️ Model dosyası boş veya geçersiz"
                st.warning(f"{self.model_hata_mesaji}\n\nLütfen model_embedded.txt dosyasını kontrol edin.")
                return
            
            try:
                model_bytes = base64.b64decode(model_base64_string)
            except Exception as decode_error:
                self.model_hata_mesaji = f"❌ Base64 decode hatası: {str(decode_error)}"
                st.error(f"{self.model_hata_mesaji}\n\nDosya içeriği Base64 formatında değil.")
                return
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(model_bytes)
                tmp_path = tmp_file.name
            
            try:
                self.lstm_model = tf.keras.models.load_model(tmp_path, compile=False)
                self.model_yuklendi = True
                
                model_size_mb = len(model_bytes) / (1024 * 1024)
                st.success(f"✅ LSTM Model başarıyla yüklendi!\n- Dosya: {os.path.basename(MODEL_PATH)}\n- Boyut: {model_size_mb:.2f} MB\n- Katman sayısı: {len(self.lstm_model.layers)}")
                
            except Exception as load_error:
                self.model_hata_mesaji = f"❌ Model yükleme hatası: {str(load_error)}"
                st.error(f"{self.model_hata_mesaji}\n\nDosya bozuk veya uyumsuz bir model olabilir.")
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
        except Exception as e:
            self.model_hata_mesaji = f"❌ Beklenmeyen hata: {str(e)}"
            st.error(self.model_hata_mesaji)
            self.model_yuklendi = False

    def csv_yukle(self, dosya, hesap_adi, baslangic_bakiye, tarih_sutun, tutar_sutun, aciklama_sutun):
        """CSV yükleme"""
        try:
            if dosya is None:
                return False, "❌ Dosya seçilmedi", pd.DataFrame()

            df = pd.read_csv(dosya, encoding='utf-8-sig')
            df.columns = df.columns.str.strip()

            if tarih_sutun not in df.columns:
                return False, f"❌ '{tarih_sutun}' sütunu bulunamadı", pd.DataFrame()

            yeni_df = pd.DataFrame()
            yeni_df['Tarih'] = pd.to_datetime(df[tarih_sutun], errors='coerce')
            yeni_df['Tutar'] = pd.to_numeric(df[tutar_sutun], errors='coerce')
            yeni_df['Aciklama'] = df[aciklama_sutun] if aciklama_sutun in df.columns else 'İşlem'

            yeni_df['Tip'] = yeni_df['Tutar'].apply(lambda x: 'Giris' if x > 0 else 'Cikis')
            yeni_df['Tutar'] = yeni_df['Tutar'].abs()
            yeni_df['Kategori'] = yeni_df['Aciklama'].apply(self._kategori_tahmin)
            yeni_df['Hesap'] = hesap_adi

            yeni_df = yeni_df.dropna(subset=['Tarih', 'Tutar'])

            if len(self.hareketler) == 0:
                self.hareketler = yeni_df.copy()
            else:
                self.hareketler = pd.concat([self.hareketler, yeni_df], ignore_index=True)

            self.hareketler = self.hareketler.sort_values('Tarih').reset_index(drop=True)
            self.hesaplar[hesap_adi] = float(baslangic_bakiye)

            return True, f"✅ {len(yeni_df)} işlem eklendi", yeni_df.head(10)

        except Exception as e:
            return False, f"❌ Hata: {str(e)}", pd.DataFrame()

    def _kategori_tahmin(self, aciklama):
        """Kategori tahmini"""
        aciklama_lower = str(aciklama).lower()
        if any(x in aciklama_lower for x in ['maas', 'maaş']):
            return 'Maaş'
        elif any(x in aciklama_lower for x in ['kira', 'rent']):
            return 'Kira'
        elif any(x in aciklama_lower for x in ['elektrik', 'su', 'gaz', 'fatura']):
            return 'Fatura'
        elif any(x in aciklama_lower for x in ['market']):
            return 'Market'
        else:
            return 'Diger'

    def manuel_islem_ekle(self, tarih, aciklama, kategori, tutar, tip, hesap):
        """Manuel işlem ekleme + OTOMATİK YENİLEME"""
        try:
            yeni_hareket = pd.DataFrame([{
                'Tarih': pd.to_datetime(tarih),
                'Aciklama': aciklama,
                'Kategori': kategori,
                'Tutar': float(tutar),
                'Tip': tip,
                'Hesap': hesap
            }])

            if len(self.hareketler) == 0:
                self.hareketler = yeni_hareket.copy()
            else:
                self.hareketler = pd.concat([self.hareketler, yeni_hareket], ignore_index=True)

            self.hareketler = self.hareketler.sort_values('Tarih').reset_index(drop=True)
            
            # 🔥 YENİ: Otomatik adaptif analizi yenile
            if self.son_analiz_parametreleri:
                self._otomatik_analiz_yenile()
            
            return True, f"✅ Eklendi: {aciklama} - {tutar} TL"
        except Exception as e:
            return False, f"❌ Hata: {str(e)}"

    def _otomatik_analiz_yenile(self):
        """Otomatik adaptif analizi yenile"""
        if self.son_analiz_parametreleri:
            params = self.son_analiz_parametreleri
            pencereler, grafik, ozet, hata = self.adaptif_analiz(
                params['baslangic_tarihi'],
                params['bitis_tarihi'],
                params['baslangic_bakiye'],
                params['buffer_tutar'],
                params['min_yatirim_tutar']
            )
            if not hata:
                self.son_adaptif_analiz = {
                    'pencereler': pencereler,
                    'grafik': grafik,
                    'ozet': ozet
                }

    # ==================== MODÜL 1: KLASİK ANALİZ ====================

    def klasik_analiz(self, baslangic_tarihi, bitis_tarihi, baslangic_bakiye, buffer_tutar):
        """Klasik nakit akış analizi"""
        try:
            if len(self.hareketler) == 0:
                return None, None, None, None, "⚠️ Veri yok!"

            baslangic = pd.to_datetime(baslangic_tarihi)
            bitis = pd.to_datetime(bitis_tarihi)

            tarih_araligi = pd.date_range(start=baslangic, end=bitis, freq='D')
            bakiye_df = pd.DataFrame({'Tarih': tarih_araligi})
            bakiye_df['Bakiye'] = float(baslangic_bakiye)

            ilgili_hareketler = self.hareketler[
                (self.hareketler['Tarih'] >= baslangic) &
                (self.hareketler['Tarih'] <= bitis)
            ].copy()

            for idx in range(len(bakiye_df)):
                tarih = bakiye_df.loc[idx, 'Tarih']

                if idx > 0:
                    bakiye_df.loc[idx, 'Bakiye'] = bakiye_df.loc[idx-1, 'Bakiye']

                gunun_hareketleri = ilgili_hareketler[
                    ilgili_hareketler['Tarih'].dt.date == tarih.date()
                ]

                for _, hareket in gunun_hareketleri.iterrows():
                    if hareket['Tip'] == 'Giris':
                        bakiye_df.loc[idx, 'Bakiye'] += hareket['Tutar']
                    else:
                        bakiye_df.loc[idx, 'Bakiye'] -= hareket['Tutar']

            bakiye_df['Yatirilabilir'] = bakiye_df['Bakiye'].apply(lambda x: max(0, x - buffer_tutar))
            bakiye_df['Durum'] = bakiye_df['Bakiye'].apply(
                lambda x: '🟢 Fazla' if x > buffer_tutar else ('🟡 Normal' if x >= 0 else '🔴 Acik')
            )

            oneriler_df = self._klasik_yatirim_onerileri(bakiye_df)
            grafik = self._klasik_grafik(bakiye_df, ilgili_hareketler)

            toplam_giris = ilgili_hareketler[ilgili_hareketler['Tip'] == 'Giris']['Tutar'].sum()
            toplam_cikis = ilgili_hareketler[ilgili_hareketler['Tip'] == 'Cikis']['Tutar'].sum()

            # 🔥 YENİ: Açık tarihleri tespit et
            acik_gunler = bakiye_df[bakiye_df['Bakiye'] < 0]
            acik_detay = []
            if len(acik_gunler) > 0:
                for _, gun in acik_gunler.iterrows():
                    acik_detay.append({
                        'Tarih': gun['Tarih'].strftime('%Y-%m-%d'),
                        'Bakiye': gun['Bakiye']
                    })

            ozet = {
                'toplam_giris': toplam_giris,
                'toplam_cikis': toplam_cikis,
                'net_akis': toplam_giris - toplam_cikis,
                'min_bakiye': bakiye_df['Bakiye'].min(),
                'max_bakiye': bakiye_df['Bakiye'].max(),
                'acik_gun': len(bakiye_df[bakiye_df['Bakiye'] < 0]),
                'acik_detay': acik_detay  # 🔥 YENİ
            }

            return bakiye_df, oneriler_df, grafik, ozet, None

        except Exception as e:
            return None, None, None, None, f"❌ Hata: {str(e)}"

    def _klasik_yatirim_onerileri(self, bakiye_df):
        """Klasik yatırım önerileri"""
        oneriler = []
        i = 0
        while i < len(bakiye_df):
            yatirilabilir = bakiye_df.iloc[i]['Yatirilabilir']

            if yatirilabilir > 50:  # 🔥 YENİ: 100'den 50'ye düşürdük
                gun_sayisi = 1
                min_tutar = yatirilabilir

                for j in range(i+1, len(bakiye_df)):
                    if bakiye_df.iloc[j]['Yatirilabilir'] > 0:
                        gun_sayisi += 1
                        min_tutar = min(min_tutar, bakiye_df.iloc[j]['Yatirilabilir'])
                    else:
                        break

                if gun_sayisi >= 1:
                    yillik_faiz = 0.45
                    gunluk_faiz = yillik_faiz / 365
                    tahmini_getiri = min_tutar * (gunluk_faiz * gun_sayisi)

                    oneriler.append({
                        'Baslangic': bakiye_df.iloc[i]['Tarih'].strftime('%Y-%m-%d'),
                        'Gun': gun_sayisi,
                        'Tutar': f"{min_tutar:,.2f} TL",
                        'Getiri': f"{tahmini_getiri:,.2f} TL",
                        'Yatirim': self._yatirim_araci_sec(gun_sayisi)
                    })

                i += gun_sayisi
            else:
                i += 1

        return pd.DataFrame(oneriler) if oneriler else pd.DataFrame()

    def _klasik_grafik(self, bakiye_df, hareketler_df):
        """Klasik grafik + NEGATİF BÖLGE VURGULU"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('💰 Günlük Bakiye', '💵 Yatırılabilir', '📊 Giriş/Çıkış'),
            vertical_spacing=0.12,
            row_heights=[0.4, 0.3, 0.3]
        )

        # 🔥 YENİ: Negatif bakiye bölgelerini kırmızı arka plan
        negatif_df = bakiye_df[bakiye_df['Bakiye'] < 0]
        for _, row in negatif_df.iterrows():
            fig.add_vrect(
                x0=row['Tarih'] - timedelta(hours=12),
                x1=row['Tarih'] + timedelta(hours=12),
                fillcolor="rgba(220, 53, 69, 0.2)",
                layer="below",
                line_width=0,
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=bakiye_df['Tarih'],
                y=bakiye_df['Bakiye'],
                mode='lines+markers',
                name='Bakiye',
                line=dict(color='#3498db', width=2),
                fill='tozeroy'
            ),
            row=1, col=1
        )

        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        fig.add_trace(
            go.Bar(
                x=bakiye_df['Tarih'],
                y=bakiye_df['Yatirilabilir'],
                name='Yatırılabilir',
                marker_color='#2ecc71'
            ),
            row=2, col=1
        )

        if len(hareketler_df) > 0:
            gunluk_giris = hareketler_df[hareketler_df['Tip'] == 'Giris'].groupby(
                hareketler_df['Tarih'].dt.date
            )['Tutar'].sum()
            gunluk_cikis = hareketler_df[hareketler_df['Tip'] == 'Cikis'].groupby(
                hareketler_df['Tarih'].dt.date
            )['Tutar'].sum()

            if len(gunluk_giris) > 0:
                fig.add_trace(go.Bar(x=gunluk_giris.index, y=gunluk_giris.values, name='Giris', marker_color='#27ae60'), row=3, col=1)

            if len(gunluk_cikis) > 0:
                fig.add_trace(go.Bar(x=gunluk_cikis.index, y=-gunluk_cikis.values, name='Cikis', marker_color='#e74c3c'), row=3, col=1)

        fig.update_layout(height=1000, showlegend=True, template='plotly_white')
        return fig

    # ==================== MODÜL 2: ADAPTİF PENCERE (GELİŞTİRİLMİŞ) ====================

    def adaptif_analiz(self, baslangic_tarihi, bitis_tarihi, baslangic_bakiye, buffer_tutar, min_yatirim_tutar):
        """Adaptif yatırım penceresi analizi - GELİŞTİRİLMİŞ"""
        try:
            if len(self.hareketler) == 0:
                return None, None, None, "⚠️ Veri yok!"

            baslangic = pd.to_datetime(baslangic_tarihi)
            bitis = pd.to_datetime(bitis_tarihi)

            tarih_araligi = pd.date_range(start=baslangic, end=bitis, freq='D')
            bakiye_df = pd.DataFrame({'Tarih': tarih_araligi})
            bakiye_df['Bakiye'] = float(baslangic_bakiye)

            ilgili_hareketler = self.hareketler[
                (self.hareketler['Tarih'] >= baslangic) &
                (self.hareketler['Tarih'] <= bitis)
            ].copy()

            for idx in range(len(bakiye_df)):
                tarih = bakiye_df.loc[idx, 'Tarih']

                if idx > 0:
                    bakiye_df.loc[idx, 'Bakiye'] = bakiye_df.loc[idx-1, 'Bakiye']

                gunun_hareketleri = ilgili_hareketler[
                    ilgili_hareketler['Tarih'].dt.date == tarih.date()
                ]

                for _, hareket in gunun_hareketleri.iterrows():
                    if hareket['Tip'] == 'Giris':
                        bakiye_df.loc[idx, 'Bakiye'] += hareket['Tutar']
                    else:
                        bakiye_df.loc[idx, 'Bakiye'] -= hareket['Tutar']

            bakiye_df['Yatirilabilir'] = bakiye_df['Bakiye'].apply(lambda x: max(0, x - buffer_tutar))

            pencereler = self._adaptif_pencereler_hesapla(bakiye_df, buffer_tutar, min_yatirim_tutar)
            
            # 🔥 YENİ: Açık tarihleri tespit et
            acik_gunler = bakiye_df[bakiye_df['Bakiye'] < 0]
            acik_detay = []
            if len(acik_gunler) > 0:
                for _, gun in acik_gunler.iterrows():
                    acik_detay.append({
                        'Tarih': gun['Tarih'].strftime('%Y-%m-%d'),
                        'Bakiye': f"{gun['Bakiye']:,.2f} TL"
                    })
            
            grafik = self._adaptif_grafik_gelismis(bakiye_df, pencereler, acik_gunler)

            ozet = {
                'pencere_sayisi': len(pencereler),
                'toplam_yatirilabilir': pencereler['Yatirilabilir_Tutar'].sum() if not pencereler.empty else 0,
                'toplam_getiri': pencereler['Tahmini_Getiri'].sum() if not pencereler.empty else 0,
                'min_bakiye': bakiye_df['Bakiye'].min(),
                'max_bakiye': bakiye_df['Bakiye'].max(),
                'acik_gun_sayisi': len(acik_gunler),  # 🔥 YENİ
                'acik_detay': acik_detay  # 🔥 YENİ
            }

            # Parametreleri kaydet (otomatik yenileme için)
            self.son_analiz_parametreleri = {
                'baslangic_tarihi': baslangic_tarihi,
                'bitis_tarihi': bitis_tarihi,
                'baslangic_bakiye': baslangic_bakiye,
                'buffer_tutar': buffer_tutar,
                'min_yatirim_tutar': min_yatirim_tutar
            }

            return pencereler, grafik, ozet, None

        except Exception as e:
            return None, None, None, f"❌ Hata: {str(e)}"

    def _adaptif_pencereler_hesapla(self, bakiye_df, buffer_tutar, min_yatirim_tutar):
        """Dinamik pencere hesaplama"""
        pencereler = []

        i = 0
        while i < len(bakiye_df):
            baslangic_tarih = bakiye_df.iloc[i]['Tarih']
            baslangic_bakiye = bakiye_df.iloc[i]['Bakiye']
            yatirilabilir_baslangic = max(0, baslangic_bakiye - buffer_tutar)

            if yatirilabilir_baslangic < min_yatirim_tutar:
                i += 1
                continue

            gun_sayisi = 0
            min_yatirilabilir = yatirilabilir_baslangic

            for j in range(i, len(bakiye_df)):
                gelecek_bakiye = bakiye_df.iloc[j]['Bakiye']
                gelecek_yatirilabilir = max(0, gelecek_bakiye - buffer_tutar)

                min_yatirilabilir = min(min_yatirilabilir, gelecek_yatirilabilir)

                if gelecek_yatirilabilir < min_yatirim_tutar or gelecek_bakiye < buffer_tutar:
                    break

                gun_sayisi += 1

            if gun_sayisi >= 1:
                bitis_tarih = baslangic_tarih + timedelta(days=gun_sayisi-1)
                
                yillik_faiz = 0.45
                gunluk_faiz = yillik_faiz / 365
                tahmini_getiri = min_yatirilabilir * (gunluk_faiz * gun_sayisi)
                
                risk = self._risk_seviyesi(bakiye_df.iloc[i:i+gun_sayisi], buffer_tutar)

                pencereler.append({
                    'Pencere_No': len(pencereler) + 1,
                    'Baslangic': baslangic_tarih.strftime('%Y-%m-%d'),
                    'Bitis': bitis_tarih.strftime('%Y-%m-%d'),
                    'Gun': gun_sayisi,
                    'Yatirilabilir_Tutar': min_yatirilabilir,
                    'Yatirim_Araci': self._yatirim_araci_sec(gun_sayisi),
                    'Tahmini_Getiri': tahmini_getiri,
                    'Yillik_Getiri': (tahmini_getiri / min_yatirilabilir) * (365 / gun_sayisi) * 100 if min_yatirilabilir > 0 else 0,
                    'Risk': risk
                })

                i += gun_sayisi
            else:
                i += 1

        return pd.DataFrame(pencereler) if pencereler else pd.DataFrame()

    def _risk_seviyesi(self, pencere_df, buffer_tutar):
        """Risk seviyesi"""
        min_bakiye = pencere_df['Bakiye'].min()
        if min_bakiye > buffer_tutar * 2:
            return "🟢 Düşük"
        elif min_bakiye > buffer_tutar * 1.5:
            return "🟡 Orta"
        elif min_bakiye > buffer_tutar:
            return "🟠 Yüksek"
        else:
            return "🔴 Kritik"

    def _yatirim_araci_sec(self, gun):
        """Yatırım aracı"""
        if gun <= 1:
            return "📊 Overnight"
        elif gun <= 7:
            return "📈 Haftalık"
        elif gun <= 30:
            return "🏦 Vadeli 1A"
        elif gun <= 90:
            return "💰 Vadeli 3A"
        else:
            return "💎 Vadeli 6A+"

    def _adaptif_grafik_gelismis(self, bakiye_df, pencereler_df, acik_gunler):
        """Adaptif grafik - GELİŞTİRİLMİŞ (Negatif bölge vurgulu)"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('💰 Bakiye ve Yatırım Pencereleri', '💵 Yatırılabilir Tutar'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )

        # 🔥 YENİ: Negatif bakiye bölgelerini kırmızı arka plan
        for _, row in acik_gunler.iterrows():
            fig.add_vrect(
                x0=row['Tarih'] - timedelta(hours=12),
                x1=row['Tarih'] + timedelta(hours=12),
                fillcolor="rgba(220, 53, 69, 0.3)",
                layer="below",
                line_width=0,
                row=1, col=1
            )
            # Uyarı ikonu ekle
            fig.add_annotation(
                x=row['Tarih'],
                y=row['Bakiye'],
                text="⚠️",
                showarrow=False,
                font=dict(size=20, color="red"),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=bakiye_df['Tarih'],
                y=bakiye_df['Bakiye'],
                mode='lines',
                name='Bakiye',
                line=dict(color='#3498db', width=2),
                fill='tozeroy'
            ),
            row=1, col=1
        )

        # 🔥 YENİ: Yatırım pencerelerini yeşil kutular
        if not pencereler_df.empty:
            for _, pencere in pencereler_df.iterrows():
                baslangic = pd.to_datetime(pencere['Baslangic'])
                bitis = pd.to_datetime(pencere['Bitis'])
                tutar = pencere['Yatirilabilir_Tutar']
                
                fig.add_shape(
                    type="rect",
                    x0=baslangic, x1=bitis,
                    y0=0, y1=tutar,
                    fillcolor="rgba(40, 167, 69, 0.3)",
                    line=dict(color="rgba(40, 167, 69, 0.7)", width=2),
                    row=1, col=1
                )
                
                fig.add_annotation(
                    x=baslangic + (bitis - baslangic) / 2,
                    y=tutar * 0.5,
                    text=f"💰 {tutar:,.0f}TL<br>{pencere['Gun']}g",
                    showarrow=False,
                    font=dict(size=9, color="green"),
                    row=1, col=1
                )

        fig.add_trace(
            go.Bar(
                x=bakiye_df['Tarih'],
                y=bakiye_df['Yatirilabilir'],
                name='Yatırılabilir',
                marker_color='#2ecc71'
            ),
            row=2, col=1
        )

        fig.update_layout(height=900, showlegend=True, template='plotly_white')
        return fig

    # ==================== MODÜL 3: LSTM TAHMİN ====================

    def lstm_tahmin(self, tahmin_gun_sayisi, baslangic_bakiye, lookback=30):
        """LSTM bakiye tahmini"""
        if not TENSORFLOW_AVAILABLE:
            return None, None, "❌ TensorFlow yüklü değil!"

        try:
            if not self.model_yuklendi or self.lstm_model is None:
                return None, None, self.model_hata_mesaji or "❌ Model yüklenemedi!"

            if len(self.hareketler) == 0:
                return None, None, "❌ Veri yok!"

            bakiye_serisi = self._gunluk_bakiye_olustur(baslangic_bakiye)

            if bakiye_serisi is None or len(bakiye_serisi) < lookback:
                return None, None, f"❌ En az {lookback} gün veri gerekli!"

            bakiye_degerleri = bakiye_serisi['Bakiye'].values.reshape(-1, 1)
            normalized_data = self.scaler.fit_transform(bakiye_degerleri)

            son_veri = normalized_data[-lookback:]

            tahminler = []
            mevcut_veri = son_veri.copy()

            for _ in range(tahmin_gun_sayisi):
                X_test = mevcut_veri.reshape(1, lookback, 1)
                tahmin = self.lstm_model.predict(X_test, verbose=0)
                tahminler.append(tahmin[0, 0])
                mevcut_veri = np.append(mevcut_veri[1:], tahmin)

            tahminler = np.array(tahminler).reshape(-1, 1)
            tahminler_gercek = self.scaler.inverse_transform(tahminler)

            son_tarih = bakiye_serisi['Tarih'].max()
            tahmin_tarihleri = pd.date_range(
                start=son_tarih + timedelta(days=1),
                periods=tahmin_gun_sayisi,
                freq='D'
            )

            tahmin_df = pd.DataFrame({
                'Tarih': tahmin_tarihleri,
                'Tahmin_Bakiye': tahminler_gercek.flatten()
            })

            grafik = self._lstm_grafik(bakiye_serisi, tahmin_df)

            ozet = {
                'tahmin_suresi': tahmin_gun_sayisi,
                'ortalama': tahminler_gercek.mean(),
                'minimum': tahminler_gercek.min(),
                'maximum': tahminler_gercek.max(),
                'risk': 'Yüksek' if tahminler_gercek.min() < 0 else 'Düşük'
            }

            return tahmin_df, grafik, ozet

        except Exception as e:
            return None, None, f"❌ Hata: {str(e)}"

    def _gunluk_bakiye_olustur(self, baslangic_bakiye):
        """Günlük bakiye serisi"""
        if len(self.hareketler) == 0:
            return None

        min_tarih = self.hareketler['Tarih'].min()
        max_tarih = self.hareketler['Tarih'].max()

        tarih_araligi = pd.date_range(start=min_tarih, end=max_tarih, freq='D')
        bakiye_serisi = pd.DataFrame({'Tarih': tarih_araligi, 'Bakiye': float(baslangic_bakiye)})

        for idx in range(len(bakiye_serisi)):
            if idx > 0:
                bakiye_serisi.loc[idx, 'Bakiye'] = bakiye_serisi.loc[idx-1, 'Bakiye']

            tarih = bakiye_serisi.loc[idx, 'Tarih']
            gunun_hareketleri = self.hareketler[self.hareketler['Tarih'].dt.date == tarih.date()]

            for _, hareket in gunun_hareketleri.iterrows():
                if hareket['Tip'] == 'Giris':
                    bakiye_serisi.loc[idx, 'Bakiye'] += hareket['Tutar']
                else:
                    bakiye_serisi.loc[idx, 'Bakiye'] -= hareket['Tutar']

        return bakiye_serisi

    def _lstm_grafik(self, gercek_veri, tahmin_veri):
        """LSTM grafik"""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=gercek_veri['Tarih'],
                y=gercek_veri['Bakiye'],
                mode='lines',
                name='Gerçek Bakiye',
                line=dict(color='#3498db', width=2)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=tahmin_veri['Tarih'],
                y=tahmin_veri['Tahmin_Bakiye'],
                mode='lines+markers',
                name='LSTM Tahmini',
                line=dict(color='#e74c3c', width=2, dash='dash'),
                marker=dict(size=6)
            )
        )

        fig.add_hline(y=0, line_dash="dot", line_color="red")

        fig.update_layout(
            title="🔮 LSTM Nakit Akış Tahmini",
            xaxis_title="Tarih",
            yaxis_title="Bakiye (TL)",
            height=600,
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

def main():
    st.title("💰 Nakit Akış Yönetim Sistemi - Tam Adaptif Versiyon")
    st.markdown("### 🎯 3 Modül: Adaptif Pencere | LSTM Tahmin | Klasik Analiz")
    st.markdown("#### 🔥 **YENİ:** Otomatik Yenileme | Açık Tarih Gösterimi | Gelişmiş Grafikler")
    
    if not st.session_state.initialized:
        st.session_state.sistem = NakitAkisYonetimiTam()
        st.session_state.initialized = True
    
    sistem = st.session_state.sistem
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Sistem Durumu")
        
        if TENSORFLOW_AVAILABLE and sistem.model_yuklendi:
            st.success("✅ LSTM Hazır")
        elif TENSORFLOW_AVAILABLE:
            st.warning("⚠️ Model Yok")
            if sistem.model_hata_mesaji:
                st.caption(sistem.model_hata_mesaji)
        else:
            st.error("❌ TensorFlow Yok")
        
        st.divider()
        
        st.caption(f"📂 Model Path:\n`{MODEL_PATH}`")
        
        st.divider()
        
        if len(sistem.hareketler) > 0:
            st.metric("📊 Toplam İşlem", len(sistem.hareketler))
            st.metric("🏦 Hesap Sayısı", len(sistem.hesaplar))
            
            st.divider()
            st.subheader("🔄 Otomatik Yenileme")
            if sistem.son_analiz_parametreleri:
                st.success("✅ Aktif")
                st.caption("Manuel işlem eklendiğinde\nadaptif analiz otomatik güncellenir")
            else:
                st.info("ℹ️ Pasif")
                st.caption("Adaptif analizi bir kez çalıştırın")
        else:
            st.info("📭 Veri yok")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Veri",
        "💰 Klasik Analiz",
        "🎯 Adaptif Pencere",
        "🔮 LSTM Tahmin",
        "📋 İşlemler",
        "❓ Yardım"
    ])
    
    # TAB 1: VERİ YÜKLE
    with tab1:
        st.header("📤 Veri Yönetimi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("CSV Yükle")
            csv_file = st.file_uploader("Dosya", type=['csv'])
            hesap_adi = st.text_input("Hesap", value="Ana Hesap")
            baslangic_bakiye = st.number_input("Başlangıç Bakiyesi", value=10000.0)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                tarih_sutun = st.text_input("Tarih Sütunu", value="Tarih")
            with col_b:
                tutar_sutun = st.text_input("Tutar Sütunu", value="Tutar")
            with col_c:
                aciklama_sutun = st.text_input("Açıklama", value="Aciklama")
            
            if st.button("📥 Yükle", type="primary", use_container_width=True):
                if csv_file:
                    success, mesaj, ozet = sistem.csv_yukle(
                        csv_file, hesap_adi, baslangic_bakiye,
                        tarih_sutun, tutar_sutun, aciklama_sutun
                    )
                    if success:
                        st.success(mesaj)
                        st.dataframe(ozet, use_container_width=True)
                    else:
                        st.error(mesaj)
        
        with col2:
            st.subheader("Manuel İşlem Ekle")
            st.info("🔥 **YENİ:** İşlem eklediğinizde adaptif analiz otomatik güncellenir!")
            
            m_tarih = st.date_input("Tarih", value=datetime.now())
            m_aciklama = st.text_input("Açıklama", placeholder="Örn: Kira ödemesi")
            m_kategori = st.selectbox("Kategori", ["Maaş", "Kira", "Fatura", "Market", "Diger"])
            
            col_d, col_e = st.columns(2)
            with col_d:
                m_tutar = st.number_input("Tutar (TL)", value=0.0, min_value=0.0)
                m_tip = st.radio("İşlem Tipi", ["Giris", "Cikis"])
            with col_e:
                m_hesap = st.text_input("Hesap", value="Kasa")
            
            if st.button("➕ İşlem Ekle", type="secondary", use_container_width=True):
                if m_aciklama and m_tutar > 0:
                    success, mesaj = sistem.manuel_islem_ekle(
                        m_tarih, m_aciklama, m_kategori, m_tutar, m_tip, m_hesap
                    )
                    if success:
                        st.success(mesaj)
                        if sistem.son_analiz_parametreleri:
                            st.info("🔄 Adaptif analiz otomatik güncellendi!")
                    else:
                        st.error(mesaj)
                else:
                    st.warning("⚠️ Açıklama ve tutar zorunludur!")
    
    # TAB 2: KLASİK ANALİZ
    with tab2:
        st.header("💰 Klasik Nakit Akış Analizi")
        
        if len(sistem.hareketler) == 0:
            st.warning("⚠️ Önce veri yükleyin")
        else:
            st.info("📌 Basit bakiye takibi ve günlük yatırılabilir tutar hesaplama")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                k_baslangic = st.date_input("Başlangıç", value=datetime.now(), key="k_bas")
            with col2:
                k_bitis = st.date_input("Bitiş", value=datetime.now() + timedelta(days=30), key="k_bit")
            with col3:
                k_bakiye = st.number_input("Bakiye", value=10000.0, key="k_bak")
            
            k_buffer = st.number_input("Buffer (TL)", value=1000.0, help="Elde tutmak istediğiniz minimum tutar", key="k_buf")
            
            if st.button("📊 KLASİK ANALİZ", type="primary", use_container_width=True):
                with st.spinner("Hesaplanıyor..."):
                    bakiye_df, oneriler, grafik, ozet, hata = sistem.klasik_analiz(
                        k_baslangic, k_bitis, k_bakiye, k_buffer
                    )
                    
                    if grafik:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Toplam Giriş", f"{ozet['toplam_giris']:,.0f} TL")
                        with col2:
                            st.metric("Toplam Çıkış", f"{ozet['toplam_cikis']:,.0f} TL")
                        with col3:
                            st.metric("Net Akış", f"{ozet['net_akis']:,.0f} TL")
                        with col4:
                            acik_gun = ozet['acik_gun']
                            st.metric("Açık Gün", acik_gun, delta="⚠️" if acik_gun > 0 else "✅")
                        
                        # 🔥 YENİ: Açık günler detayı
                        if ozet['acik_detay']:
                            st.markdown("---")
                            st.subheader("⚠️ Negatif Bakiye Uyarıları")
                            for acik in ozet['acik_detay']:
                                st.markdown(f"""
                                <div class="warning-card">
                                    <strong>🔴 {acik['Tarih']}</strong>: Bakiye {acik['Bakiye']:,.2f} TL
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.plotly_chart(grafik, use_container_width=True)
                        
                        if not oneriler.empty:
                            st.subheader("💎 Yatırım Önerileri")
                            st.dataframe(oneriler, use_container_width=True)
                    else:
                        st.error(hata)
    
    # TAB 3: ADAPTİF PENCERE
    with tab3:
        st.header("🎯 Adaptif Yatırım Penceresi")
        
        if len(sistem.hareketler) == 0:
            st.warning("⚠️ Önce veri yükleyin")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                a_baslangic = st.date_input("Başlangıç", value=datetime.now(), key="a_bas")
            with col2:
                a_bitis = st.date_input("Bitiş", value=datetime.now() + timedelta(days=30), key="a_bit")
            with col3:
                a_bakiye = st.number_input("Başlangıç Bakiyesi", value=10000.0, key="a_bak")
            with col4:
                a_buffer = st.number_input("Buffer Tutarı", value=1000.0, key="a_buf")
            
            a_min_yatirim = st.slider("Minimum Yatırım (TL)", 10, 500, 50, 10, 
                                      help="🔥 YENİ: 50 TL'ye düşürüldü - küçük tutarlar da gösterilir")
            
            if st.button("🎯 ADAPTİF ANALİZ", type="primary", use_container_width=True):
                with st.spinner("Pencereler hesaplanıyor..."):
                    pencereler, grafik, ozet, hata = sistem.adaptif_analiz(
                        a_baslangic, a_bitis, a_bakiye, a_buffer, a_min_yatirim
                    )
                    
                    if grafik:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Pencere Sayısı", ozet['pencere_sayisi'])
                        with col2:
                            st.metric("Toplam Yatırılabilir", f"{ozet['toplam_yatirilabilir']:,.0f} TL")
                        with col3:
                            st.metric("Tahmini Getiri", f"{ozet['toplam_getiri']:,.2f} TL")
                        with col4:
                            acik = ozet['acik_gun_sayisi']
                            st.metric("Açık Gün", acik, delta="⚠️" if acik > 0 else "✅")
                        
                        # 🔥 YENİ: Açık günler detayı
                        if ozet['acik_detay']:
                            st.markdown("---")
                            st.subheader("⚠️ Negatif Bakiye Uyarıları")
                            cols = st.columns(3)
                            for idx, acik in enumerate(ozet['acik_detay']):
                                with cols[idx % 3]:
                                    st.markdown(f"""
                                    <div class="warning-card">
                                        <strong>🔴 {acik['Tarih']}</strong><br>
                                        Bakiye: {acik['Bakiye']}
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        st.plotly_chart(grafik, use_container_width=True)
                        
                        if not pencereler.empty:
                            st.subheader("💎 Yatırım Pencereleri")
                            for _, p in pencereler.iterrows():
                                st.markdown(f"""
                                <div class="adaptive-card" style="background-color: black; color: white; padding: 15px; border-radius: 5px;">
                                    <h4>Pencere #{p['Pencere_No']}: {p['Baslangic']} → {p['Bitis']}</h4>
                                    <p>💰 <strong>{p['Yatirilabilir_Tutar']:,.2f} TL</strong> | ⏱️ {p['Gun']} gün | 📊 {p['Yatirim_Araci']}</p>
                                    <p>💵 Getiri: {p['Tahmini_Getiri']:,.2f} TL (Yıllık: %{p['Yillik_Getiri']:.2f}) | ⚠️ Risk: {p['Risk']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            display_df = pencereler.copy()
                            display_df['Yatirilabilir_Tutar'] = display_df['Yatirilabilir_Tutar'].apply(lambda x: f"{x:,.2f}")
                            display_df['Tahmini_Getiri'] = display_df['Tahmini_Getiri'].apply(lambda x: f"{x:,.2f}")
                            display_df['Yillik_Getiri'] = display_df['Yillik_Getiri'].apply(lambda x: f"%{x:.2f}")
                            st.dataframe(display_df, use_container_width=True)
                        else:
                            st.info("💡 Pencere bulunamadı. Buffer veya min yatırım tutarını azaltın.")
                    else:
                        st.error(hata)
    
    # TAB 4: LSTM TAHMİN
    with tab4:
        st.header("🔮 LSTM Bakiye Tahmini")
        
        if not TENSORFLOW_AVAILABLE:
            st.error("❌ TensorFlow yüklü değil!")
        elif not sistem.model_yuklendi:
            st.warning("⚠️ Model yüklenemedi.")
            if sistem.model_hata_mesaji:
                st.error(sistem.model_hata_mesaji)
            st.info("💡 Model dosyasını kontrol edin ve sayfayı yenileyin.")
        elif len(sistem.hareketler) == 0:
            st.warning("⚠️ Önce veri yükleyin")
        else:
            st.markdown("""

            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                l_tahmin_gun = st.slider("Tahmin Süresi (Gün)", 1, 90, 30)
            with col2:
                l_bakiye = st.number_input("Başlangıç Bakiyesi", value=10000.0, key="l_bak")
            with col3:
                l_lookback = st.slider("Lookback Period", 7, 60, 30)
            
            if st.button("🔮 LSTM TAHMİN", type="primary", use_container_width=True):
                with st.spinner("AI hesaplıyor..."):
                    tahmin_df, grafik, ozet = sistem.lstm_tahmin(
                        l_tahmin_gun, l_bakiye, l_lookback
                    )
                    
                    if isinstance(ozet, dict):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Tahmin Süresi", f"{ozet['tahmin_suresi']} gün")
                        with col2:
                            st.metric("Ort. Bakiye", f"{ozet['ortalama']:,.0f} TL")
                        with col3:
                            st.metric("Min Bakiye", f"{ozet['minimum']:,.0f} TL")
                        with col4:
                            risk_color = "🔴" if ozet['risk'] == 'Yüksek' else "🟢"
                            st.metric("Açık Riski", f"{risk_color} {ozet['risk']}")
                        
                        st.plotly_chart(grafik, use_container_width=True)
                        
                        st.subheader("📋 Tahmin Detayları")
                        display_df = tahmin_df.copy()
                        display_df['Tarih'] = display_df['Tarih'].dt.strftime('%Y-%m-%d')
                        display_df['Tahmin_Bakiye'] = display_df['Tahmin_Bakiye'].apply(lambda x: f"{x:,.2f}")
                        display_df['Durum'] = tahmin_df['Tahmin_Bakiye'].apply(
                            lambda x: '🟢 Pozitif' if x >= 0 else '🔴 Negatif'
                        )
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.error(ozet)
    
    # TAB 5: TÜM İŞLEMLER
    with tab5:
        st.header("📋 Tüm İşlemler")
        
        if len(sistem.hareketler) == 0:
            st.info("Henüz işlem yok")
        else:
            df = sistem.hareketler.copy()
            df['Tarih'] = df['Tarih'].dt.strftime('%Y-%m-%d')
            df = df.sort_values('Tarih', ascending=False)
            
            st.dataframe(df, use_container_width=True, height=600)
            
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 CSV İndir",
                data=csv,
                file_name=f"nakit_akis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # TAB 6: YARDIM
    with tab6:
        st.header("❓ Kullanım Kılavuzu - YENİ ÖZELLİKLER")
        
        st.markdown(f"""
        ## 🔥 Yeni Özellikler (v2.0)
        
        ### 1. ✅ Otomatik Yenileme
        **Sorun:** Manuel işlem eklenince "Adaptif Analiz" butonuna tekrar basmak gerekiyordu
        **Çözüm:** Artık işlem eklediğinizde analiz otomatik güncellenir!
        
        **Nasıl Çalışır:**
        1. Adaptif Analizi bir kez çalıştırın
        2. Manuel işlem ekleyin
        3. ✨ Analiz otomatik yenilenir!
        
        ---
        
        ### 2. ⚠️ Açık Tarih Gösterimi
        **Sorun:** Sadece "5 gün açık" gösteriyordu, hangi tarihler belli değildi
        **Çözüm:** Artık negatif bakiye olan tarihleri görebilirsiniz!
        
        **Örnek Çıktı:**
        ```
        ⚠️ Negatif Bakiye Uyarıları
        🔴 2025-01-15: Bakiye -500 TL
        🔴 2025-01-16: Bakiye -300 TL
        ```
        
        ---
        
        ### 3. 📊 Gelişmiş Grafikler
        **Yeni:** Negatif bakiye bölgeleri kırmızı arka plan ile vurgulanır
        
        **Grafik Özellikleri:**
        - 🟢 Yeşil kutular = Yatırım pencereleri
        - 🔴 Kırmızı arka plan = Negatif bakiye riski
        - ⚠️ İkonlar = Açık günler
        
        ---
        
        ### 4. 📉 Minimum Yatırım 50 TL'ye Düştü
        **Eski:** 100 TL altı gösterilmiyordu
        **Yeni:** 50 TL'den başlıyor - küçük tutarlar da önerilir
        
        **Örnek:**
        - "5-31 Ocak: 50 TL (26 gün)" ✅ Artık gösteriliyor
        
        ---
        
        ## 🎯 Tam Senaryo Örneği
        
        **Durum:**
        - 1 Ocak: 1000 TL bakiye
        - 1 Ocak: 50 TL çıkış → Kalan: 950 TL
        - 5 Ocak: 900 TL ödeme → Kalan: 50 TL
        - Buffer: 0 TL
        
        **Sistem Çıktısı:**
        ```
        Pencere #1: 2025-01-01 → 2025-01-04
        💰 950 TL | ⏱️ 4 gün | 📊 Haftalık
        💵 Getiri: 4.67 TL
        
        Pencere #2: 2025-01-05 → 2025-01-31
        💰 50 TL | ⏱️ 26 gün | 📊 Vadeli 1A
        💵 Getiri: 1.60 TL
        ```
        
        ---
        
        ## 📚 Sistem Modülleri
        
        ### 1️⃣ Klasik Analiz
        - ✅ Basit bakiye takibi
        - ✅ Günlük yatırılabilir tutar
        - ✅ Giriş/çıkış grafiği
        - 🔥 **YENİ:** Açık tarih gösterimi
        - 🎯 **Kullanım:** Hızlı bakış için
        
        ### 2️⃣ Adaptif Yatırım Penceresi ⭐ ÖNERİLEN
        - ✅ Dinamik pencere hesaplama
        - ✅ Her gün için: kaç gün, ne kadar?
        - ✅ Risk analizi
        - ✅ Optimum strateji
        - 🔥 **YENİ:** Otomatik yenileme
        - 🔥 **YENİ:** Açık tarih vurgulama
        - 🔥 **YENİ:** 50 TL'den başlayan öneriler
        - 🎯 **Kullanım:** Profesyonel yatırım planı için
        
        ### 3️⃣ LSTM Tahmin
        - ✅ AI bazlı bakiye tahmini
        - ✅ Gelecek projeksiyonu
        - ✅ Risk uyarısı
        - 🎯 **Kullanım:** Uzun vadeli planlama için
        
        ---
        
        ## 🚀 Kullanım Adımları
        
        ### Başlangıç
        1. **Veri Yükle** → CSV veya manuel işlem ekle
        2. **Adaptif Analiz** → Pencereler hesapla
        3. **Otomatik Mod Aktif** → Artık yeni işlem eklediğinde otomatik güncellenir
        
        ### Manuel İşlem Eklerken
        ```
        Tarih: 2025-01-15
        Açıklama: Kira ödemesi
        Tutar: 1500 TL
        Tip: Çıkış
        
        [İşlem Ekle] ✅
        → 🔄 "Adaptif analiz otomatik güncellendi!"
        ```
        
        ---
        
        ## 💡 Hangi Modülü Kullanmalıyım?
        
        | Durum | Önerilen Modül |
        |-------|----------------|
        | Hızlı bakiye kontrolü | Klasik Analiz |
        | **Yatırım stratejisi** | **Adaptif Pencere** ⭐ |
        | Gelecek planlaması | LSTM Tahmin |
        | Kapsamlı analiz | Adaptif + LSTM |
        | Açık risk tespiti | Klasik veya Adaptif (ikisinde de var) |
        
        ---
        
        ## 🔧 Sık Sorulan Sorular
        
        **S: "1-4 Ocak: 950 TL" neden 5 Ocak'a kadar değil?**
        C: 5 Ocak'ta 900 TL ödeme var, bu yüzden pencere 4'te kapanır. Buffer'ın altına düşmesin diye.
        
        **S: Otomatik yenileme çalışmıyor?**
        C: Adaptif Analizi en az bir kez çalıştırmış olmalısınız. Sidebar'da "✅ Aktif" yazmalı.
        
        **S: 50 TL'lik pencere gösterilmiyor?**
        C: "Minimum Yatırım" slider'ını 50 TL'nin altına çekin.
        
        **S: Açık tarihler nerede görünüyor?**
        C: Hem Klasik hem Adaptif analizde "⚠️ Negatif Bakiye Uyarıları" başlığı altında.
        
        **S: Grafikteki kırmızı alanlar ne?**
        C: Negatif bakiye olan (açık veren) günlerdir.
        
        **S: Buffer tutarı ne olmalı?**
        C: Aylık giderlerinizin 1-2 katı önerilir. Örnek: Aylık 5000 TL gider → 5000-10000 TL buffer.
        
        **S: Adaptif vs Klasik farkı?**
        C: 
        - **Klasik:** "Bugün X TL yatırılabilir" (tek günlük)
        - **Adaptif:** "1-5 Ocak: X TL, 5-31 Ocak: Y TL" (pencere bazlı)
        
        ---
        
        ## 📁 LSTM Model Kurulumu
        
        ### Mevcut Konfigürasyon:
        ```python
        MODEL_PATH = "{MODEL_PATH}"
        ```
        
        ### Model Hazırlama:
        ```python
        import base64
        
        # .h5 modelini Base64'e çevir
        with open('nakit_akis_lstm_final.h5', 'rb') as f:
            model_bytes = f.read()
            model_base64 = base64.b64encode(model_bytes).decode('utf-8')
        
        # Text dosyasına kaydet
        with open('model_embedded.txt', 'w') as f:
            f.write(model_base64)
        
        print("✅ model_embedded.txt oluşturuldu!")
        ```
        
        ### Google Colab'da:
        ```python
        from google.colab import files
        uploaded = files.upload()  # model_embedded.txt seç
        
        # Path doğru: /content/model_embedded.txt
        ```
        
        ### Yerel'de:
        ```python
        MODEL_PATH = "model_embedded.txt"  # Aynı klasörde
        # veya
        MODEL_PATH = "C:/Users/Name/models/model_embedded.txt"  # Tam yol
        ```
        
        ---
        
        ## 📊 CSV Format Örneği
        
        ```csv
        Tarih,Tutar,Aciklama
        2024-10-01,5000,Maaş
        2024-10-02,-1200,Kira
        2024-10-05,-300,Market
        2024-10-10,2000,Proje Ödemesi
        2024-10-15,-500,Fatura
        ```
        
        **Kurallar:**
        - Pozitif tutar = Giriş
        - Negatif tutar = Çıkış
        - Tarih: YYYY-MM-DD formatı
        
        ---
        
        ## 🎨 Yatırım Araçları
        
        | Gün Sayısı | Araç | Örnek |
        |------------|------|-------|
        | 1 gün | 📊 Overnight | Günlük repo |
        | 2-7 gün | 📈 Haftalık | Haftalık repo |
        | 8-30 gün | 🏦 Vadeli 1A | 1 aylık vadeli |
        | 31-90 gün | 💰 Vadeli 3A | 3 aylık vadeli |
        | 90+ gün | 💎 Vadeli 6A+ | 6 aylık+ vadeli |
        
        **Faiz Hesabı:** Yıllık %45 → Günlük %0.123
        
        ---
        
        ## 🚀 En İyi Pratikler
        
        1. **Düzenli Veri Güncelleme**
           - Her hafta yeni işlemleri ekleyin
           - Otomatik yenileme aktif olsun
        
        2. **Buffer Stratejisi**
           - Minimum: Aylık gider × 1
           - Önerilen: Aylık gider × 1.5-2
           - Güvenli: Aylık gider × 2-3
        
        3. **Risk Yönetimi**
           - 🟢 Düşük risk pencerelerini tercih edin
           - 🔴 Kritik risk varsa buffer'ı artırın
           - Negatif bakiye uyarılarını ciddiye alın
        
        4. **Modül Kombinasyonu**
           - **İlk analiz:** Adaptif Pencere
           - **Doğrulama:** Klasik Analiz
           - **Gelecek:** LSTM Tahmin
        
        5. **Model Kalitesi**
           - LSTM için en az 60-90 gün veri
           - Her 3 ayda bir modeli yeniden eğitin
           - Lookback period: 30 gün (optimal)
        
        ---
        
        ## 🔄 Değişiklik Geçmişi
        
        ### v2.0 (Mevcut)
        - ✅ Otomatik yenileme eklendi
        - ✅ Açık tarih gösterimi eklendi
        - ✅ Grafiklerde negatif bölge vurgusu
        - ✅ Minimum yatırım 50 TL'ye düşürüldü
        - ✅ Detaylı hata mesajları
        
        ### v1.0 (Eski)
        - Temel adaptif pencere analizi
        - LSTM tahmin modülü
        - Klasik bakiye takibi
        
        ---
        
        ## 📞 Destek
        
        **Kod Yapısı:**
        - Modüler: LSTM olmadan da çalışır
        - Hata yönetimi: Her adımda try-catch
        - Performans: Verimli pandas operasyonları
        
        **Uyumluluk:**
        - ✅ Google Colab
        - ✅ Yerel Python
        - ✅ Streamlit Cloud
        - ✅ Windows/Mac/Linux
        
        ---
        
        ## 💪 Sistem Avantajları
        
        ✅ **Adaptif:** Her işlemde otomatik güncelleme
        ✅ **Görsel:** Kırmızı/yeşil bölge vurguları
        ✅ **Detaylı:** Açık tarihleri gösterir
        ✅ **Akıllı:** 50 TL'den başlar, her tutar için öneri
        ✅ **Güvenli:** Buffer sistemi ile risk yönetimi
        ✅ **Esnek:** 3 farklı analiz modülü
        
        ---
        
        ## 🎯 Sonuç
        
        Bu sistem artık **tam adaptif**! İstediğiniz gibi:
        - ✅ Otomatik yenileme
        - ✅ Açık tarih gösterimi
        - ✅ Negatif bakiye vurgusu
        - ✅ Küçük tutarlar için öneri
        
        **Kullanmaya başlayın:** 
        1. Veri Sekmesinden CSV yükleyin
        2. Adaptif Pencere'yi çalıştırın
        3. Manuel işlem ekleyin
        4. 🔄 Otomatik güncellemeyi izleyin!
        
        **İyi yatırımlar!** 💰
        """)

if __name__ == "__main__":
    main()
