from flask import Flask, render_template, request, send_file
import os, cv2, numpy as np, sqlite3, datetime, json
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
REPORT_FOLDER = "reports"
DB = "database.db"

for f in [UPLOAD_FOLDER, RESULT_FOLDER, REPORT_FOLDER]:
    os.makedirs(f, exist_ok=True)

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        tool TEXT,
        result TEXT,
        timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

def save_experiment(filename, tool, result):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO experiments (filename, tool, result, timestamp) VALUES (?, ?, ?, ?)",
              (filename, tool, json.dumps(result), str(datetime.datetime.now())))
    conn.commit()
    conn.close()

def get_experiments():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM experiments ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return data

# ================= BLOOD =================
def process_image(image_path, output_path, heatmap_path):
    img = cv2.imread(image_path)
    original = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0,(8,8)).apply(gray)

    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,2)

    sure_bg = cv2.dilate(opening,kernel,3)

    dist = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    _, sure_fg = cv2.threshold(dist,0.3*dist.max(),255,0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)

    rbc,wbc = 0,0

    for label in np.unique(markers):
        if label <= 1: continue
        mask = np.zeros(gray.shape,dtype="uint8")
        mask[markers==label] = 255

        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 5000:
                if area < 800:
                    rbc+=1; color=(0,255,0)
                else:
                    wbc+=1; color=(255,0,0)
                cv2.drawContours(original,[cnt],-1,color,2)

    total = rbc+wbc

    heatmap = cv2.applyColorMap(gray,cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_path,heatmap)

    cv2.putText(original,f"RBC:{rbc}",(20,40),0,1,(0,255,0),2)
    cv2.putText(original,f"WBC:{wbc}",(20,80),0,1,(255,0,0),2)
    cv2.putText(original,f"Total:{total}",(20,120),0,1,(0,0,255),2)

    cv2.imwrite(output_path,original)

    return {"rbc":rbc,"wbc":wbc,"total":total}

# ================= COLONY =================
def analyze_colonies(image_path, output_path, heatmap_path):
    img = cv2.imread(image_path)
    original = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0,(8,8)).apply(gray)

    blur = cv2.GaussianBlur(gray,(7,7),0)
    _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,2)

    dist = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    _, sure_fg = cv2.threshold(dist,0.4*dist.max(),255,0)

    sure_fg = np.uint8(sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers = cv2.watershed(img,markers)

    count=0; sizes=[]
    for label in np.unique(markers):
        if label <= 1: continue
        mask = np.zeros(gray.shape,dtype="uint8")
        mask[markers==label] = 255

        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 20000:
                count+=1
                sizes.append(area)
                cv2.drawContours(original,[cnt],-1,(0,255,0),2)

    avg = int(np.mean(sizes)) if sizes else 0

    heatmap = cv2.applyColorMap(gray,cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_path,heatmap)

    cv2.putText(original,f"Colonies:{count}",(20,40),0,1,(0,255,0),2)
    cv2.putText(original,f"Avg:{avg}",(20,80),0,1,(255,0,0),2)

    cv2.imwrite(output_path,original)

    return {"count":count,"avg":avg}

# ================= COMPOUND =================
def screen_compounds(csv_path, output_path):
    df = pd.read_csv(csv_path)
    results = []
    passed = []

    for _, row in df.iterrows():
        score = 0

        if row['mw'] <= 500: score += 1
        if row['logp'] <= 5: score += 1
        if row['hbd'] <= 5: score += 1
        if row['hba'] <= 10: score += 1

        status = "PASS" if score >= 3 else "FAIL"

        result = {
            "name": row['name'],
            "score": score,
            "status": status
        }

        results.append(result)

        if status == "PASS":
            passed.append(result)

    pd.DataFrame(results).to_csv(output_path, index=False)

    pass_path = os.path.join(RESULT_FOLDER, "passed_compounds.csv")
    pd.DataFrame(passed).to_csv(pass_path, index=False)

    return results

# ================= ROUTES =================
@app.route("/",methods=["GET","POST"])
def index():
    results=[]
    if request.method=="POST":
        tool=request.form.get("tool")
        files=request.files.getlist("images")

        for file in files:
            if file.filename=="": continue
            name=secure_filename(file.filename)

            up=os.path.join(UPLOAD_FOLDER,name)
            res=os.path.join(RESULT_FOLDER,name)
            heat=os.path.join(RESULT_FOLDER,"heat_"+name)

            file.save(up)

            if tool=="blood":
                out=process_image(up,res,heat)
            else:
                out=analyze_colonies(up,res,heat)

            save_experiment(name,tool,out)

            results.append({"name":name,"tool":tool,"result":out,"heat":"heat_"+name})

    return render_template("index.html",results=results)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html",data=get_experiments())

@app.route("/screen",methods=["GET","POST"])
def screen():
    results=None
    if request.method=="POST":
        file=request.files["file"]
        path=os.path.join(UPLOAD_FOLDER,file.filename)
        out=os.path.join(RESULT_FOLDER,"screen.csv")
        file.save(path)
        results=screen_compounds(path,out)
    return render_template("screen.html",results=results)

@app.route("/uploads/<f>")
def up(f): return send_file(os.path.join(UPLOAD_FOLDER,f))

@app.route("/results/<f>")
def res(f): return send_file(os.path.join(RESULT_FOLDER,f))

@app.route("/download_passed")
def download_passed():
    path = os.path.join(RESULT_FOLDER, "passed_compounds.csv")
    return send_file(path, as_attachment=True)
    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
