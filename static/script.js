const logsEl = document.getElementById("logs");
const resultsEl = document.getElementById("results");
const uploadForm = document.getElementById("uploadForm");
const searchForm = document.getElementById("searchForm");

function appendLog(msg) {
    logsEl.textContent += msg + "\n";
    logsEl.scrollTop = logsEl.scrollHeight;
}

uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const filesInput = document.getElementById("files");
    if (filesInput.files.length === 0) {
        alert("אנא בחר קבצים");
        return;
    }
    appendLog("מתחיל העלאת קבצים...");
    const formData = new FormData();
    for (const file of filesInput.files) {
        formData.append("files", file);
    }
    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        appendLog(data.message);
    } catch (err) {
        appendLog("שגיאה בהעלאת הקבצים");
        console.error(err);
    }
});

searchForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = document.getElementById("query").value.trim();
    const apiKey = document.getElementById("apiKey").value.trim();
  

    if (!query) {
        alert("אנא הזן שאלה");
        return;
    }
    if (!apiKey) {
        alert("אנא הזן את מפתח ה-OpenAI API");
        return;
    }

    appendLog("מבצע חיפוש...");
    resultsEl.innerHTML = ""; // נקה תוצאות קודמות

    try {
        const response = await fetch(`/search?q=${encodeURIComponent(query)}&apikey=${encodeURIComponent(apiKey)}`);
        const data = await response.json();

        if (data.error) {
            appendLog("שגיאה: " + data.error);
            return;
        }

        appendLog("תוצאות החיפוש:");
        data.results.forEach((res, i) => {
            const div = document.createElement("div");
            div.className = "result-item";

            const textDiv = document.createElement("div");
            textDiv.className = "result-text";
            textDiv.textContent = `${i + 1}. ${res.text}`;
            div.appendChild(textDiv);

            if (res.image_paths && res.image_paths.length > 0) {
                const imagesDiv = document.createElement("div");
                imagesDiv.className = "result-images";
                res.image_paths.forEach(src => {
                    const img = document.createElement("img");
                    img.src = src;
                    imagesDiv.appendChild(img);
                });
                div.appendChild(imagesDiv);
            }

            resultsEl.appendChild(div);
        });

        if (data.answer) {
            const answerDiv = document.createElement("div");
            answerDiv.className = "result-item";
            answerDiv.style.backgroundColor = "#e0f7fa";
            answerDiv.style.border = "2px solid #00796b";
            answerDiv.style.fontWeight = "bold";
            answerDiv.textContent = "תשובת המודל:\n" + data.answer;
            resultsEl.appendChild(answerDiv);
        }
    } catch (err) {
        appendLog("שגיאה בחיפוש");
        console.error(err);
    }
});
