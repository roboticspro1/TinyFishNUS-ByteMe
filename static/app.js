const analyzeBtn = document.getElementById("analyzeBtn");
const urlInput = document.getElementById("url");
const goalInput = document.getElementById("goal");
const schemaOutput = document.getElementById("schemaOutput");
const jsonOutput = document.getElementById("jsonOutput");
const apiOutput = document.getElementById("apiOutput");
const modeBadge = document.getElementById("modeBadge");
const summaryCards = document.getElementById("summaryCards");
const siteMap = document.getElementById("siteMap");
const schemaCards = document.getElementById("schemaCards");
const sampleCards = document.getElementById("sampleCards");
const agentNotes = document.getElementById("agentNotes");
const explanationText = document.getElementById("explanationText");
const curlExample = document.getElementById("curlExample");
const pythonExample = document.getElementById("pythonExample");
const jsExample = document.getElementById("jsExample");
const sampleTableHead = document.getElementById("sampleTableHead");
const sampleTableBody = document.getElementById("sampleTableBody");
const cardsView = document.getElementById("cardsView");
const tableView = document.getElementById("tableView");
const rawView = document.getElementById("rawView");

document.querySelectorAll(".preset").forEach((button) => {
    button.addEventListener("click", () => {
        urlInput.value = button.dataset.url;
        goalInput.value = button.dataset.goal;
    });
});

document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".tab-button").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".code-example").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.target).classList.add("active");
    });
});

document.querySelectorAll(".view-button").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".view-button").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        cardsView.classList.add("hidden");
        tableView.classList.add("hidden");
        rawView.classList.add("hidden");
        document.getElementById(button.dataset.view).classList.remove("hidden");
    });
});

function renderSummary(result) {
    summaryCards.innerHTML = `
        <article class="summary-item">
            <span class="summary-label">Entity</span>
            <strong>${result.site.entity_guess}</strong>
        </article>
        <article class="summary-item">
            <span class="summary-label">Confidence</span>
            <strong>${Math.round(result.site.confidence * 100)}%</strong>
        </article>
        <article class="summary-item">
            <span class="summary-label">Endpoints</span>
            <strong>${result.generated_api.endpoints.length}</strong>
        </article>
        <article class="summary-item">
            <span class="summary-label">Page classes</span>
            <strong>${result.site_map.length}</strong>
        </article>
    `;
}

function renderSiteMap(result) {
    siteMap.innerHTML = result.site_map.map((page) => `
        <div class="stack-card">
            <strong>${page.page_type}</strong>
            <div>${page.notes}</div>
            <div class="sample-meta"><span>${page.pattern}</span></div>
        </div>
    `).join("");
}

function renderSchema(result) {
    schemaCards.innerHTML = result.schema.fields.map((field) => `
        <div class="field-card">
            <strong>${field.name}</strong>
            <span>${field.type}</span>
        </div>
    `).join("");
    schemaOutput.textContent = JSON.stringify(result.schema, null, 2);
}

function renderSamples(result) {
    cardsView.innerHTML = result.samples.slice(0, 4).map((item) => {
        const meta = [];
        if (item.author) meta.push(`<span>${item.author}</span>`);
        if (item.published_at) meta.push(`<span>${item.published_at}</span>`);
        if (item.location) meta.push(`<span>${item.location}</span>`);
        if (item.price_text) meta.push(`<span>${item.price_text}</span>`);

        return `
            <article class="sample-card">
                <strong>${item.title || "Untitled"}</strong>
                <p>${item.summary || item.url || "No summary available."}</p>
                <div class="sample-meta">${meta.join("")}</div>
            </article>
        `;
    }).join("");
    rawView.textContent = JSON.stringify(result.samples, null, 2);

    const columns = [...new Set(result.samples.flatMap((item) => Object.keys(item)))];
    sampleTableHead.innerHTML = `<tr>${columns.map((column) => `<th>${column}</th>`).join("")}</tr>`;
    sampleTableBody.innerHTML = result.samples.slice(0, 8).map((item) => `
        <tr>
            ${columns.map((column) => {
                const value = item[column] ?? "";
                if (column === "url" && value) {
                    return `<td><a href="${value}" target="_blank" rel="noreferrer">Open Link</a></td>`;
                }
                return `<td>${String(value)}</td>`;
            }).join("")}
        </tr>
    `).join("");
}

function renderNotes(result) {
    agentNotes.innerHTML = result.agent_notes.map((note) => `
        <div class="stack-card">${note}</div>
    `).join("");
}

function renderQuickstart(result) {
    curlExample.textContent = result.quickstart.curl;
    pythonExample.textContent = result.quickstart.python;
    jsExample.textContent = result.quickstart.javascript;
}

function renderExplanation(result) {
    explanationText.textContent = `Web2API Studio analyzed ${result.site.hostname} as a likely ${result.site.entity_guess} source. It inferred a reusable schema, identified ${result.site_map.length} page classes, and designed an API surface developers can use without manually studying the site's HTML.`;
}

async function analyzeWebsite() {
    const payload = {
        url: urlInput.value.trim(),
        goal: goalInput.value.trim(),
    };

    if (!payload.url || !payload.goal) {
        modeBadge.textContent = "Missing input";
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing...";
    modeBadge.textContent = "Agent running";
    explanationText.textContent = "Exploring the target website, grouping repeated pages, and inferring a reusable schema.";
    schemaOutput.textContent = "Inspecting structural patterns and building a schema candidate...";
    rawView.textContent = "Collecting representative records...";
    apiOutput.textContent = "Generating developer-ready endpoint design...";
    siteMap.innerHTML = `<div class="stack-card">Clustering possible landing, listing, and detail page patterns...</div>`;
    schemaCards.innerHTML = `<div class="field-card"><strong>Analyzing</strong><span>Deriving fields from extracted records.</span></div>`;
    cardsView.innerHTML = `<div class="sample-card"><strong>Collecting records</strong><p>Readable sample cards will appear here.</p></div>`;
    sampleTableHead.innerHTML = "";
    sampleTableBody.innerHTML = "";
    agentNotes.innerHTML = `<div class="stack-card">Preparing the reverse-engineering workflow...</div>`;
    renderQuickstart({
        quickstart: {
            curl: "Generating cURL example...",
            python: "Generating Python example...",
            javascript: "Generating JavaScript example...",
        },
    });

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || "Analysis failed.");
        }

        renderSummary(result);
        renderSiteMap(result);
        renderSchema(result);
        renderSamples(result);
        renderNotes(result);
        renderQuickstart(result);
        renderExplanation(result);
        modeBadge.textContent = `${result.mode} mode`;
        apiOutput.textContent = JSON.stringify(result.generated_api, null, 2);
    } catch (error) {
        modeBadge.textContent = "Error";
        explanationText.textContent = "The analysis request failed. Check that the local server is running and try again.";
        schemaOutput.textContent = "The analysis request failed.";
        rawView.textContent = error.message;
        apiOutput.textContent = "Please retry after checking the local server.";
        siteMap.innerHTML = `<div class="stack-card">No site map available because the analysis request failed.</div>`;
        schemaCards.innerHTML = `<div class="field-card"><strong>No schema</strong><span>The analysis did not complete.</span></div>`;
        cardsView.innerHTML = `<div class="sample-card"><strong>No sample data</strong><p>The analysis did not complete.</p></div>`;
        sampleTableHead.innerHTML = "";
        sampleTableBody.innerHTML = "";
        agentNotes.innerHTML = `<div class="stack-card">Unable to complete the reverse-engineering workflow.</div>`;
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analyze Website";
    }
}

analyzeBtn.addEventListener("click", analyzeWebsite);
