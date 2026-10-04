"use strict";

(function () {
  const form = document.querySelector("#analyze-form");
  const overlay = document.querySelector("#checking-overlay");
  const button = document.querySelector("#analyze-button");

  if (form && overlay) {
    form.addEventListener("submit", function () {
      overlay.hidden = false;
      if (button) {
        button.disabled = true;
        button.textContent = "Checking resumes...";
      }
    });
  }

  const search = document.querySelector("#candidate-search");
  const filter = document.querySelector("#candidate-filter");
  const sort = document.querySelector("#candidate-sort");
  const cards = Array.from(document.querySelectorAll(".candidate-card"));
  const empty = document.querySelector("#no-filter-results");

  function refreshCards() {
    const query = (search ? search.value : "").trim().toLowerCase();
    const filterValue = filter ? filter.value : "all";

    cards.forEach(function (card) {
      const haystack = (card.dataset.search || "").toLowerCase();
      const score = Number(card.dataset.score || "0");
      const textMatch = !query || haystack.includes(query);
      const filterMatch =
        filterValue === "all" ||
        (filterValue === "strong" && score >= 70) ||
        (filterValue === "review" && score >= 50 && score < 70) ||
        (filterValue === "low" && score < 50);
      card.hidden = !(textMatch && filterMatch);
    });

    if (empty) {
      empty.hidden = cards.some(function (card) { return !card.hidden; });
    }
  }

  function sortCards() {
    const container = document.querySelector("#candidate-list");
    if (!container || !sort) return;

    const direction = sort.value;
    const sorted = cards.slice().sort(function (a, b) {
      const scoreA = Number(a.dataset.score || "0");
      const scoreB = Number(b.dataset.score || "0");
      const nameA = (a.dataset.name || "").toLowerCase();
      const nameB = (b.dataset.name || "").toLowerCase();
      if (direction === "name") return nameA.localeCompare(nameB);
      return direction === "low" ? scoreA - scoreB : scoreB - scoreA;
    });

    sorted.forEach(function (card) { container.appendChild(card); });
    refreshCards();
  }

  if (search) search.addEventListener("input", refreshCards);
  if (filter) filter.addEventListener("change", refreshCards);
  if (sort) sort.addEventListener("change", sortCards);
})();
