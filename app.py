from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import mannwhitneyu

# AquaBio Explorer | Đồ án Data Mining sinh viên
st.set_page_config(page_title="AquaBio Explorer", page_icon="🌊", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
import base64

# ========== VIDEO BACKGROUND ==========
VIDEO_PATH = BASE_DIR / "assets" / "abcde.mp4"

if VIDEO_PATH.exists():
    with open(VIDEO_PATH, "rb") as video_file:
        video_base64 = base64.b64encode(
            video_file.read()
        ).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background: transparent;
    }}

    .video-background {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 0;
        overflow: hidden;
        pointer-events: none;
    }}

    .video-background video {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}

    .video-background::after {{
        content: "";
        position: absolute;
        inset: 0;
        background: rgba(3, 15, 35, 0.60);
    }}

    [data-testid="stAppViewContainer"] {{
        background: transparent;
    }}
    </style>

    <div class="video-background">
        <video autoplay muted loop playsinline>
            <source src="data:video/mp4;base64,{video_base64}"
                    type="video/mp4">
        </video>
    </div>
    """, unsafe_allow_html=True)
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
FILES = {
    "s001": "bdj-11-e105314-s001.csv",
    "s002": "bdj-11-e105314-s002.csv",
    "s003": "bdj-11-e105314-s003.csv",
    "s004": "bdj-11-e105314-s004.csv",
}
JOIN_KEYS = ["Trawl", "Waterbody", "Datetime"]
SAMPLE_KEYS = ["Trawl", "Waterbody", "Datetime", "Depth"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#071923;color:#e8f4f2}
[data-testid="stHeader"]{background:rgba(7,25,35,.9)}
[data-testid="stSidebar"]{background:#0b2632;border-right:1px solid #1d4652}
.hero{background:linear-gradient(120deg,#0c3440,#0a2633 55%,#123f49);border:1px solid #24515b;border-radius:24px;padding:30px;margin:5px 0 22px}
.hero small{color:#79d9c4;letter-spacing:.18em;font-weight:800}
.hero h1{color:#f2fffb;font-family:Manrope,sans-serif;font-size:46px;margin:10px 0}
.hero p{color:#b5d4d2;font-size:16px;margin:0}
.section-title{color:#f1fbf8;font-family:Manrope,sans-serif;font-size:25px;font-weight:800;margin:18px 0 5px}
.note{background:#103944;border:1px solid #28515a;border-radius:14px;padding:14px;color:#d9eeea}
[data-testid="stMetric"]{background:#103744;border:1px solid #24515b;padding:14px;border-radius:14px}
[data-testid="stMetricLabel"]{color:#9cc5c4!important}[data-testid="stMetricValue"]{color:#f1fbf8!important}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_csv_smart(path_string):
    path=Path(path_string)
    errors=[]
    for enc in ("utf-8-sig","utf-8","cp1252","latin1"):
        for sep in (";",",","\t"):
            try:
                df=pd.read_csv(path,sep=sep,encoding=enc)
                if df.shape[1]>1:
                    df.columns=[str(c).strip() for c in df.columns]
                    return df
            except Exception as exc:
                errors.append(str(exc))
    raise ValueError(f"Không đọc được {path.name}. Kiểm tra định dạng CSV.")

@st.cache_data(show_spinner=False)
def load_data(data_dir):
    result={}
    for key,name in FILES.items():
        p=Path(data_dir)/name
        if not p.exists():
            raise FileNotFoundError(f"Thiếu {p}. Hãy đặt đủ bốn CSV vào thư mục data.")
        result[key]=read_csv_smart(str(p))
    return result

def make_sample_data(biology):
    required=set(SAMPLE_KEYS+['Taxa','Number'])
    missing=required-set(biology.columns)
    if missing:
        raise ValueError('s003 thiếu cột: '+', '.join(sorted(missing)))
    bio=biology.copy()
    bio['Number']=pd.to_numeric(bio['Number'],errors='coerce')
    if (bio['Number'].dropna()<0).any():
        raise ValueError('Number có giá trị âm; cần kiểm tra dữ liệu gốc.')
    if 'Biomass' in bio:
        bio['Biomass']=pd.to_numeric(bio['Biomass'],errors='coerce')
    else:
        bio['Biomass']=np.nan
    samples=(bio.groupby(SAMPLE_KEYS,dropna=False)
        .agg(Taxa_Richness=('Taxa','nunique'),Total_Number=('Number',lambda x:x.sum(min_count=1)),
             Total_Biomass=('Biomass',lambda x:x.sum(min_count=1)),
             Mean_Number=('Number','mean'),Mean_Biomass=('Biomass','mean')).reset_index())
    valid_rows=bio.dropna(subset=['Number','Taxa'])
    wide=valid_rows.pivot_table(index=SAMPLE_KEYS,columns='Taxa',values='Number',aggfunc='sum',fill_value=0)
    wide.columns=wide.columns.astype(str); wide.columns.name=None
    samples=samples.set_index(SAMPLE_KEYS)
    wide=wide.reindex(samples.index,fill_value=0)
    samples=samples.reset_index()
    wide=wide.reset_index()
    meta=wide[SAMPLE_KEYS].copy()
    taxa_cols=[c for c in wide.columns if c not in SAMPLE_KEYS]
    matrix=wide[taxa_cols].apply(pd.to_numeric,errors='coerce').fillna(0).clip(lower=0)
    totals=matrix.sum(axis=1)
    valid=totals>0
    relative=matrix.loc[valid].div(totals[valid],axis=0)
    hellinger=np.sqrt(relative)
    return bio,samples,wide,meta,taxa_cols,valid,relative,hellinger

def attach_environment(samples, environment):
    if not set(JOIN_KEYS).issubset(environment.columns):
        return samples.copy(), False
    env=environment.copy()
    duplicate=env.duplicated(JOIN_KEYS,keep=False)
    if duplicate.any():
        # Tổng hợp các bản ghi cùng chuyến: mean cho biến số, first cho biến phân loại.
        numeric=env.select_dtypes(include=np.number).columns.tolist()
        categorical=[c for c in env.columns if c not in JOIN_KEYS+numeric]
        agg={c:'mean' for c in numeric}; agg.update({c:'first' for c in categorical})
        env=env.groupby(JOIN_KEYS,dropna=False).agg(agg).reset_index()
    merged=samples.merge(env,on=JOIN_KEYS,how='left',validate='many_to_one',suffixes=('_bio','_env'),indicator=True)
    return merged, True

def evaluate_matrix(matrix, k, seed=42):
    model=KMeans(n_clusters=int(k),random_state=seed,n_init=30)
    labels=model.fit_predict(matrix)
    if len(np.unique(labels))<2 or len(np.unique(labels))>=len(matrix):
        return None
    return {'model':model,'labels':labels,'silhouette':silhouette_score(matrix,labels),
            'db':davies_bouldin_score(matrix,labels),'ch':calinski_harabasz_score(matrix,labels),
            'inertia':model.inertia_}

def cluster_pipeline(matrix, max_k=10):
    max_k=min(max_k,len(matrix)-1)
    if max_k<2: raise ValueError('Cần tối thiểu 3 mẫu hợp lệ để phân cụm.')
    rows=[]
    for k in range(2,max_k+1):
        r=evaluate_matrix(matrix,k)
        if r: rows.append({'K':k,'Silhouette':r['silhouette'],'Davies_Bouldin':r['db'],
                           'Calinski_Harabasz':r['ch'],'Inertia':r['inertia']})
    result=pd.DataFrame(rows)
    if result.empty: raise ValueError('Không tính được chỉ số đánh giá cho các K đã thử.')
    best_k=int(result.loc[result.Silhouette.idxmax(),'K'])
    final=evaluate_matrix(matrix,best_k)
    return result,best_k,final

def bh_adjust(p_values):
    p=np.asarray(p_values,dtype=float); n=len(p)
    order=np.argsort(p); ranked=p[order]
    adjusted=np.minimum.accumulate((ranked*n/np.arange(1,n+1))[::-1])[::-1]
    out=np.empty(n); out[order]=np.clip(adjusted,0,1)
    return out

try:
    data=load_data(str(DATA_DIR))
    df_water,df_environment,df_biology,df_sediment=(data['s001'],data['s002'],data['s003'],data['s004'])
    bio,samples,abundance,meta,taxa_cols,valid_mask,relative,hellinger=make_sample_data(df_biology)
    sample_env,has_env=attach_environment(samples,df_environment)
except Exception as exc:
    st.error(str(exc))
    st.info('Cấu trúc thư mục: app.py, data/ và bốn tệp CSV s001–s004.')
    st.stop()

st.sidebar.markdown('## 🌊 AquaBio Explorer')
waterbodies=sorted(samples['Waterbody'].dropna().astype(str).unique()) if 'Waterbody' in samples else []
selected_waterbody=st.sidebar.selectbox('Waterbody', ['Tất cả']+waterbodies)
max_k=min(10,len(hellinger)-1)
if max_k<2:
    st.error('Không đủ mẫu hợp lệ để phân cụm (cần ít nhất 3).'); st.stop()
max_k_ui=st.sidebar.slider('K tối đa dùng khi khảo sát',2,max(2,max_k),min(8,max_k))
st.sidebar.caption('K-Means sử dụng abundance Taxa biến đổi Hellinger. Biến môi trường chỉ dùng để diễn giải sau phân cụm.')

st.markdown('<div class="hero"><small>AQUATIC BIODIVERSITY · DATA MINING</small><h1>AquaBio Explorer</h1><p>Khám phá cấu trúc quần xã thủy sinh và tìm hiểu sự khác biệt về điều kiện môi trường giữa các nhóm mẫu.</p></div>',unsafe_allow_html=True)

quality=[]
for name,df in data.items():
    quality.append({'Dataset':name,'Số dòng':len(df),'Số cột':df.shape[1],
                    'Missing (%)':round(df.isna().sum().sum()/max(df.size,1)*100,2),
                    'Dòng trùng':int(df.duplicated().sum())})
qdf=pd.DataFrame(quality)
valid_n=int(valid_mask.sum())
col1,col2,col3,col4=st.columns(4)
col1.metric('Bản ghi sinh học',f'{len(bio):,}')
col2.metric('Mẫu khảo sát',f'{len(samples):,}')
col3.metric('Taxa',f'{len(taxa_cols):,}')
col4.metric('Mẫu đưa vào phân cụm',f'{valid_n:,}')

st.markdown('<div class="section-title">01 · Tổng quan dữ liệu</div>',unsafe_allow_html=True)
st.dataframe(qdf,use_container_width=True,hide_index=True)
with st.expander('Xem tỷ lệ thiếu theo từng biến'):
    for name,df in data.items():
        st.markdown(f'**{name.upper()}**')
        miss=(df.isna().mean()*100).sort_values(ascending=False)
        st.dataframe(miss[miss>0].rename('Missing (%)').to_frame().round(2),use_container_width=True)

st.markdown('<div class="section-title">02 · Khám phá dữ liệu (EDA)</div>',unsafe_allow_html=True)
view_samples=samples.copy()
if selected_waterbody!='Tất cả' and 'Waterbody' in view_samples:
    view_samples=view_samples[view_samples.Waterbody.astype(str)==selected_waterbody]
left,right=st.columns(2)
with left:
    if 'Waterbody' in view_samples:
        counts=view_samples['Waterbody'].astype(str).value_counts().rename_axis('Waterbody').reset_index(name='Số mẫu')
        st.plotly_chart(px.bar(counts,x='Waterbody',y='Số mẫu',color='Số mẫu',title='Số mẫu theo Waterbody'),use_container_width=True)
with right:
    bio_view=bio.copy()
    if selected_waterbody!='Tất cả': bio_view=bio_view[bio_view.Waterbody.astype(str)==selected_waterbody]
    top=bio_view.groupby('Taxa')['Number'].sum(min_count=1).nlargest(12).reset_index()
    st.plotly_chart(px.bar(top,x='Number',y='Taxa',orientation='h',title='12 Taxa có tổng Number cao nhất'),use_container_width=True)

st.markdown('**Điều kiện môi trường – s002**')
env_num=[c for c in df_environment.select_dtypes(include=np.number).columns if c not in ['Station','Latitude','Longitude','ELongitude','ELatitude','SLongitude','SLatitude']]
if env_num:
    env_choice=st.selectbox('Chọn chỉ tiêu môi trường',env_num,key='env_eda')
    env_plot=df_environment.copy()
    if selected_waterbody!='Tất cả' and 'Waterbody' in env_plot:
        env_plot=env_plot[env_plot.Waterbody.astype(str)==selected_waterbody]
    if 'Waterbody' in env_plot:
        st.plotly_chart(px.box(env_plot,x='Waterbody',y=env_choice,points='outliers',title=f'{env_choice} theo Waterbody'),use_container_width=True)
    st.caption('Các biểu đồ EDA mô tả phân phối dữ liệu, không hàm ý quan hệ nhân quả.')

st.markdown('**Trầm tích – s004**')
sed_num=df_sediment.select_dtypes(include=np.number).columns.tolist()
if sed_num:
    sed_choice=st.selectbox('Chọn biến trầm tích',sed_num,index=sed_num.index('Hg') if 'Hg' in sed_num else 0)
    st.plotly_chart(px.histogram(df_sediment,x=sed_choice,nbins=25,title=f'Phân phối {sed_choice}'),use_container_width=True)

st.markdown('<div class="section-title">03 · Tiền xử lý và lựa chọn biểu diễn quần xã</div>',unsafe_allow_html=True)
st.markdown('<div class="note">Mỗi mẫu được xác định bởi Trawl, Waterbody, Datetime và Depth. Ma trận mẫu × Taxa chứa tổng Number. Mẫu có tổng abundance bằng 0 không được đưa vào phân cụm. Giá trị Number bị thiếu được giữ là thiếu ở bước kiểm tra và không tự động xem là số 0.</div>',unsafe_allow_html=True)
representation=st.selectbox('Biểu diễn dữ liệu dùng cho K-Means',['Hellinger','Relative abundance','Raw abundance'])
# Tạo các biểu diễn trên cùng tập mẫu hợp lệ.
raw_matrix=abundance[taxa_cols].apply(pd.to_numeric,errors='coerce').fillna(0).clip(lower=0).loc[valid_mask].reset_index(drop=True)
rel_matrix=raw_matrix.div(raw_matrix.sum(axis=1),axis=0)
hell_matrix=np.sqrt(rel_matrix)
matrices={'Raw abundance':raw_matrix,'Relative abundance':rel_matrix,'Hellinger':hell_matrix}
selected_matrix=matrices[representation]

# Bộ lọc Waterbody áp dụng thực sự vào phân cụm.
valid_meta=meta.loc[valid_mask].reset_index(drop=True)
if selected_waterbody!='Tất cả':
    subset=valid_meta['Waterbody'].astype(str)==selected_waterbody
    selected_matrix=selected_matrix.loc[subset].reset_index(drop=True)
    valid_meta=valid_meta.loc[subset].reset_index(drop=True)
if len(selected_matrix)<3:
    st.warning('Waterbody đang chọn có ít hơn 3 mẫu hợp lệ. Hãy chọn Tất cả hoặc khu vực khác.'); st.stop()

st.markdown('<div class="section-title">04 · Phân cụm K-Means</div>',unsafe_allow_html=True)
with st.spinner('Đang khảo sát số cụm K...'):
    try:
        k_table,best_k,final=cluster_pipeline(selected_matrix,max_k_ui)
    except Exception as exc:
        st.error(f'Không thể phân cụm: {exc}'); st.stop()
st.caption(f'K được đề xuất là {best_k}, tương ứng Silhouette cao nhất trong khoảng K = 2 đến {max_k_ui}. Có thể chọn K khác để xem kết quả.')
chosen_k=st.slider('Chọn K để xem kết quả',2,max_k_ui,best_k)
chosen=evaluate_matrix(selected_matrix,chosen_k)
labels=chosen['labels']
metric1,metric2,metric3=st.columns(3)
metric1.metric('Silhouette',f"{chosen['silhouette']:.3f}")
metric2.metric('Davies–Bouldin',f"{chosen['db']:.3f}")
metric3.metric('Calinski–Harabasz',f"{chosen['ch']:.2f}")

c1,c2=st.columns(2)
with c1:
    st.plotly_chart(px.line(k_table,x='K',y='Inertia',markers=True,title='Elbow Method – Inertia'),use_container_width=True)
with c2:
    st.plotly_chart(px.line(k_table,x='K',y='Silhouette',markers=True,title='Silhouette theo K'),use_container_width=True)
st.dataframe(k_table.round(4),use_container_width=True,hide_index=True)

pca=PCA(n_components=2,random_state=42)
coords=pca.fit_transform(selected_matrix)
plot_df=valid_meta.copy(); plot_df['Cluster']=labels.astype(str); plot_df['PC1']=coords[:,0]; plot_df['PC2']=coords[:,1]
plot_df['Sample']=plot_df['Trawl'].astype(str)+' · '+plot_df['Waterbody'].astype(str)
st.plotly_chart(px.scatter(plot_df,x='PC1',y='PC2',color='Cluster',hover_name='Sample',hover_data=['Depth'],title=f'PCA projection – {representation}',opacity=.8),use_container_width=True)
st.caption(f'Hai thành phần PCA giải thích {pca.explained_variance_ratio_.sum()*100:.1f}% phương sai. PCA chỉ phục vụ trực quan hóa; nhãn cụm được tạo trong không gian đặc trưng ban đầu.')

# Cluster profile trên abundance tương đối cùng thứ tự sample đã lọc.
cluster_rel=rel_matrix.copy()
if selected_waterbody!='Tất cả': cluster_rel=cluster_rel.loc[subset].reset_index(drop=True)
cluster_rel['Cluster']=labels
sizes=pd.Series(labels).value_counts().sort_index().rename_axis('Cluster').reset_index(name='Số mẫu')
st.plotly_chart(px.pie(sizes,names='Cluster',values='Số mẫu',hole=.55,title='Tỷ lệ số mẫu theo cụm'),use_container_width=True)
profile=cluster_rel.groupby('Cluster')[taxa_cols].mean()
st.markdown('**Taxa có abundance tương đối trung bình cao theo cụm**')
for cid,row in profile.iterrows():
    st.write(f'Cluster {cid}')
    st.dataframe(row.sort_values(ascending=False).head(10).rename('Mean relative abundance').to_frame().style.format('{:.3%}'),use_container_width=True)

st.markdown('<div class="section-title">05 · Diễn giải môi trường hậu nghiệm</div>',unsafe_allow_html=True)
st.markdown('<div class="note">Các biến môi trường không được sử dụng để tạo cụm. Chúng chỉ được ghép sau đó để mô tả và so sánh các cụm. Kiểm định Mann–Whitney U được dùng cho hai cụm; p-value được hiệu chỉnh Benjamini–Hochberg khi kiểm định nhiều biến. Đây là phân tích liên hệ, không chứng minh quan hệ nhân quả.</div>',unsafe_allow_html=True)
if has_env:
    # gắn nhãn cụm đúng với các mẫu đã được dùng trong phân cụm
    clustered_meta=valid_meta.copy(); clustered_meta['Cluster']=labels
    env_for_test,_=attach_environment(clustered_meta,df_environment)
    env_candidates=[c for c in ['Depth_env','Length','TSpeed','TDuration','s.T','b.T','s.S','b.S','s.DOX','b.DOX','s.pH','b.pH','PlastD','PlantD'] if c in env_for_test.columns]
    env_candidates += [c for c in env_for_test.select_dtypes(include=np.number).columns if c not in env_candidates+['Depth_bio'] and c not in SAMPLE_KEYS]
    env_candidates=list(dict.fromkeys(env_candidates))
    rows=[]
    if len(np.unique(labels)) == 2:
      for col in env_candidates:
          vals=[]
          for cid in sorted(np.unique(labels)):
              vals.append(pd.to_numeric(env_for_test.loc[env_for_test.Cluster==cid,col],errors='coerce').dropna())
          if len(vals)==2 and len(vals[0])>=2 and len(vals[1])>=2:
              stat,p=mannwhitneyu(vals[0],vals[1],alternative='two-sided')
              rows.append({'Biến':col,'N cụm 0':len(vals[0]),'Median cụm 0':vals[0].median(),
                           'N cụm 1':len(vals[1]),'Median cụm 1':vals[1].median(),'p-value':p})
    if rows:
        test_df=pd.DataFrame(rows); test_df['p-value hiệu chỉnh (BH)']=bh_adjust(test_df['p-value'])
        test_df=test_df.sort_values('p-value hiệu chỉnh (BH)')
        st.dataframe(test_df.round(4),use_container_width=True,hide_index=True)
        env_choice=st.selectbox('Xem phân phối biến môi trường',test_df['Biến'].tolist())
        env_for_test[env_choice]=pd.to_numeric(env_for_test[env_choice],errors='coerce')
        st.plotly_chart(px.box(env_for_test,x='Cluster',y=env_choice,points='outliers',title=f'{env_choice} theo cụm'),use_container_width=True)
    else:
        st.info('Chưa có biến môi trường đủ số quan sát ở cả hai cụm để kiểm định.')
    if len(np.unique(labels)) != 2:
        st.info('Kiểm định Mann–Whitney hiện áp dụng cho K = 2; với K khác 2, xem hồ sơ mô tả và biểu đồ theo cụm.')
else:
    st.info('Không tìm thấy đủ khóa để ghép s002 và s003.')

st.markdown('<div class="section-title">06 · Tải kết quả</div>',unsafe_allow_html=True)
export_labels=plot_df.copy()
st.download_button('Tải nhãn cụm CSV',export_labels.to_csv(index=False).encode('utf-8-sig'),'aquabio_cluster_labels.csv','text/csv')
st.download_button('Tải hồ sơ Taxa CSV',profile.to_csv().encode('utf-8-sig'),'aquabio_taxa_profile.csv','text/csv')
st.download_button('Tải bảng chất lượng dữ liệu CSV',qdf.to_csv(index=False).encode('utf-8-sig'),'aquabio_data_quality.csv','text/csv')

with st.expander('Thông tin phương pháp và giới hạn'):
    st.markdown("""
    - K-Means là thuật toán phân cụm không giám sát cơ bản; K được khảo sát bằng Elbow và Silhouette.
    - Raw abundance có thể bị ảnh hưởng bởi quy mô tổng cá thể; Relative abundance và Hellinger tập trung hơn vào thành phần tương đối.
    - Silhouette càng cao càng tốt; Davies–Bouldin càng thấp càng tốt; Calinski–Harabasz càng cao càng tốt. Các chỉ số chỉ đánh giá cấu trúc nội tại.
    - Các ô 0 trong ma trận được hiểu là không có bản ghi Taxa tương ứng. Cần kiểm tra metadata gốc để xác nhận ý nghĩa này.
    - Dòng Number thiếu không được tự động xem là quan sát bằng 0. Mẫu có tổng abundance hợp lệ bằng 0 được loại khỏi phân cụm.
    - Dữ liệu môi trường có missing; mỗi kiểm định sử dụng số quan sát hợp lệ được hiển thị trong bảng.
    - Kết quả phụ thuộc dữ liệu khảo sát, cách tổng hợp mẫu và ngưỡng lọc; không suy rộng quá phạm vi dữ liệu.
    """)
st.caption('AquaBio Explorer · Final Data Mining Project')
