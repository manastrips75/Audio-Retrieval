// =============================================
// Upload Media
// =============================================

const uploadBtn = document.getElementById("uploadBtn");
const mediaFile = document.getElementById("mediaFile");

uploadBtn.addEventListener("click", async () => {

    if (mediaFile.files.length === 0) {

        alert("Please choose a media file.");

        return;

    }

    const formData = new FormData();

    formData.append(

        "media",

        mediaFile.files[0]

    );

    uploadBtn.disabled = true;

    uploadBtn.innerText = "Uploading...";

    try {

        const response = await fetch(

            "/upload-media",

            {

                method: "POST",

                body: formData,

            }

        );

        const result = await response.json();

        alert(result.message);

    }

    catch (err) {

        alert("Upload failed.");

    }

    uploadBtn.disabled = false;

    uploadBtn.innerText = "Upload";

});


// =============================================
// Build Index
// =============================================

const buildBtn = document.getElementById("buildBtn");

buildBtn.addEventListener("click", async () => {

    buildBtn.disabled = true;

    buildBtn.innerText = "Building...";

    try {

        await fetch(

            "/build-index",

            {

                method: "POST",

            }

        );

    }

    catch (err) {

        alert("Unable to start indexing.");

        buildBtn.disabled = false;

        buildBtn.innerText = "Build Index";

    }

});


// =============================================
// Progress Polling
// =============================================

setInterval(async () => {

    try {

        const response = await fetch("/progress");

        const data = await response.json();

        document.getElementById("status").innerHTML =
            data.status || "-";

        document.getElementById("currentFile").innerHTML =
            data.file || "-";

        document.getElementById("chunk").innerHTML =
            (data.chunk || 0) +
            " / " +
            (data.total_chunks || 0);

        document.getElementById("eta").innerHTML =
            data.eta || "--:--";

        document.getElementById("speed").innerHTML =
            (data.speed || "-") + " chunks/sec";

        const bar =
            document.getElementById("progressBar");

        bar.style.width =
            (data.progress || 0) + "%";

        bar.innerHTML =
            Math.round(data.progress || 0) + "%";

        if (!data.running) {

            buildBtn.disabled = false;

            buildBtn.innerText = "Build Index";

        }

    }

    catch (err) {

        // Ignore polling errors

    }

}, 500);


// =============================================
// Search
// =============================================

const searchBtn = document.getElementById("searchBtn");
const queryFile = document.getElementById("queryFile");

searchBtn.addEventListener("click", async () => {

    if (queryFile.files.length === 0) {

        alert("Please choose a query audio.");

        return;

    }

    const formData = new FormData();

    // IMPORTANT
    formData.append(

        "query_audio",

        queryFile.files[0]

    );

    searchBtn.disabled = true;

    searchBtn.innerText = "Searching...";

    try {

        const response = await fetch(

            "/search",

            {

                method: "POST",

                body: formData,

            }

        );

        const result = await response.json();

        searchBtn.disabled = false;

        searchBtn.innerText = "Search";

        if (!result.success) {

            alert(result.message);

            return;

        }

        document.getElementById("resultFile").innerHTML =
            result.file;

        document.getElementById("resultStart").innerHTML =
            result.start;

        document.getElementById("resultEnd").innerHTML =
            result.end;

        document.getElementById("resultSimilarity").innerHTML =
            result.similarity;

    }

    catch (err) {

        searchBtn.disabled = false;

        searchBtn.innerText = "Search";

        alert("Search failed.");

    }

});