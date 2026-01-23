// 파일: static/js/loan_check.js
document.getElementById("checkBtn").addEventListener("click", async () => {
    const loanId = document.getElementById("loan_id").value.trim();
    const resultCard = document.getElementById("resultCard");
    const resultDiv = document.getElementById("result");
    const personalDiv = document.getElementById("personalInfo");

    if (!loanId) {
        alert("Loan ID를 입력하세요.");
        return;
    }

    resultCard.classList.remove("d-none");
    resultDiv.innerHTML = "조회 중...";
    personalDiv.innerHTML = "";

    try {
        const response = await fetch("/predict_by_loan_id", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ loan_id: loanId })
        });

        if (!response.ok) {
            const err = await response.json();
            resultDiv.innerHTML = `<span class="text-danger">오류: ${err.detail}</span>`;
            return;
        }

        const data = await response.json();
        resultDiv.innerHTML = `
            승인 여부: <strong>${data.approved ? "승인 가능" : "승인 불가"}</strong><br>
            승인 확률: <strong>${(data.predict_loan * 100).toFixed(2)}%</strong>
        `;

        let infoHtml = "";
        for (const [key, value] of Object.entries(data.personal_info)) {
            infoHtml += `${key}: ${value}<br>`;
        }
        personalDiv.innerHTML = infoHtml;

    } catch (err) {
        resultDiv.innerHTML = `<span class="text-danger">서버 통신 실패: ${err}</span>`;
    }
});

