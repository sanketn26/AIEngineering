/**
 * AI Engineering Course — static gamification for MkDocs / GitHub Pages.
 * Progress lives in localStorage only (no backend required).
 */
(function () {
  "use strict";

  var STORAGE_KEY = "aieng-progress-v1";
  var XP_PER_LEVEL = 250;

  var MODULES = [
    { id: "01", title: "Prompt engineering", xp: 100 },
    { id: "02", title: "Security & privacy", xp: 100 },
    { id: "03", title: "Advanced prompting", xp: 100 },
    { id: "04", title: "Testing & evals", xp: 100 },
    { id: "05", title: "Context engineering", xp: 120 },
    { id: "06", title: "Fine-tuning", xp: 120 },
    { id: "07", title: "Tools & basic RAG", xp: 120 },
    { id: "08", title: "Model Context Protocol", xp: 100 },
    { id: "09", title: "Advanced RAG", xp: 120 },
    { id: "10", title: "Cost optimization", xp: 80 },
    { id: "11", title: "Single agents", xp: 120 },
    { id: "12", title: "Multi-agent systems", xp: 120 },
    { id: "13", title: "Production", xp: 120 },
    { id: "14", title: "Compliance", xp: 80 },
    { id: "15", title: "Domain apps", xp: 80 },
    { id: "16", title: "Integration patterns", xp: 100 },
    { id: "17", title: "Small & local models", xp: 100 }
  ];

  var BADGES = [
    {
      id: "first-steps",
      icon: "🚀",
      name: "First Steps",
      desc: "Earn your first XP",
      test: function (s) {
        return s.xp > 0;
      }
    },
    {
      id: "quiz-rookie",
      icon: "🧠",
      name: "Quiz Rookie",
      desc: "Answer 1 quiz correctly",
      test: function (s) {
        return Object.keys(s.quizzes || {}).length >= 1;
      }
    },
    {
      id: "quiz-ace",
      icon: "🎯",
      name: "Quiz Ace",
      desc: "Answer 10 quizzes correctly",
      test: function (s) {
        return Object.keys(s.quizzes || {}).length >= 10;
      }
    },
    {
      id: "module-one",
      icon: "📘",
      name: "Module Complete",
      desc: "Finish any core module",
      test: function (s) {
        return Object.keys(s.modules || {}).length >= 1;
      }
    },
    {
      id: "foundations",
      icon: "🧱",
      name: "Foundations",
      desc: "Complete modules 01–04",
      test: function (s) {
        return ["01", "02", "03", "04"].every(function (id) {
          return s.modules && s.modules[id];
        });
      }
    },
    {
      id: "retrieval-pro",
      icon: "🔎",
      name: "Retrieval Pro",
      desc: "Complete modules 07 and 09",
      test: function (s) {
        return s.modules && s.modules["07"] && s.modules["09"];
      }
    },
    {
      id: "agent-ops",
      icon: "🤖",
      name: "Agent Ops",
      desc: "Complete modules 11 and 12",
      test: function (s) {
        return s.modules && s.modules["11"] && s.modules["12"];
      }
    },
    {
      id: "ship-it",
      icon: "🚢",
      name: "Ship It",
      desc: "Complete Production (13)",
      test: function (s) {
        return s.modules && s.modules["13"];
      }
    },
    {
      id: "security-mindset",
      icon: "🛡️",
      name: "Security Mindset",
      desc: "Complete Security (02)",
      test: function (s) {
        return s.modules && s.modules["02"];
      }
    },
    {
      id: "half-stack",
      icon: "⚡",
      name: "Half Stack",
      desc: "Complete 9 core modules",
      test: function (s) {
        return Object.keys(s.modules || {}).length >= 9;
      }
    },
    {
      id: "full-core",
      icon: "🏆",
      name: "Full Core",
      desc: "Complete all 17 core modules",
      test: function (s) {
        return Object.keys(s.modules || {}).length >= 17;
      }
    },
    {
      id: "level-5",
      icon: "⭐",
      name: "Level 5",
      desc: "Reach player level 5",
      test: function (s) {
        return levelFromXp(s.xp || 0) >= 5;
      }
    }
  ];

  function defaultState() {
    return {
      xp: 0,
      modules: {},
      quizzes: {},
      badges: {},
      thinks: {},
      updatedAt: null
    };
  }

  function load() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return defaultState();
      var parsed = JSON.parse(raw);
      return Object.assign(defaultState(), parsed);
    } catch (e) {
      return defaultState();
    }
  }

  function save(state) {
    state.updatedAt = new Date().toISOString();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }

  function levelFromXp(xp) {
    return Math.floor(xp / XP_PER_LEVEL) + 1;
  }

  function xpIntoLevel(xp) {
    return xp % XP_PER_LEVEL;
  }

  function toast(msg) {
    var el = document.getElementById("aieng-toast");
    if (!el) {
      el = document.createElement("div");
      el.id = "aieng-toast";
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(el._t);
    el._t = setTimeout(function () {
      el.classList.remove("show");
    }, 2600);
  }

  function awardBadges(state, silent) {
    var newly = [];
    BADGES.forEach(function (b) {
      if (state.badges[b.id]) return;
      if (b.test(state)) {
        state.badges[b.id] = new Date().toISOString();
        newly.push(b);
      }
    });
    if (!silent && newly.length) {
      newly.forEach(function (b) {
        toast("Badge unlocked: " + b.icon + " " + b.name);
      });
    }
    return newly;
  }

  function addXp(state, amount, reason) {
    if (!amount) return;
    state.xp += amount;
    awardBadges(state, false);
    save(state);
    renderHud(state);
    if (reason) toast("+" + amount + " XP — " + reason);
  }

  function moduleById(id) {
    for (var i = 0; i < MODULES.length; i++) {
      if (MODULES[i].id === id) return MODULES[i];
    }
    return null;
  }

  function detectModuleId() {
    var meta = document.querySelector("[data-module-id]");
    if (meta && meta.getAttribute("data-module-id")) {
      return meta.getAttribute("data-module-id");
    }
    var path = (location.pathname || "").toLowerCase();
    var m = path.match(/\/(\d{2})[-_]/);
    if (m) return m[1];
    m = path.match(/(\d{2})-[a-z0-9-]+/);
    return m ? m[1] : null;
  }

  function renderHud(state) {
    var hud = document.getElementById("aieng-hud");
    if (!hud) {
      hud = document.createElement("div");
      hud.id = "aieng-hud";
      hud.className = "collapsed";
      hud.innerHTML =
        '<div class="aieng-hud-head" id="aieng-hud-toggle">' +
        '<span class="aieng-hud-title">AI Eng Progress</span>' +
        '<span id="aieng-hud-chevron">▸</span></div>' +
        '<div class="aieng-hud-body">' +
        '<div class="aieng-row"><span>Level</span><strong id="aieng-level">1</strong></div>' +
        '<div class="aieng-row"><span>XP</span><strong id="aieng-xp">0</strong></div>' +
        '<div class="aieng-bar"><span id="aieng-bar"></span></div>' +
        '<div class="aieng-row"><span>Modules</span><strong id="aieng-mods">0/17</strong></div>' +
        '<div class="aieng-badges" id="aieng-hud-badges"></div>' +
        '<div class="aieng-links">' +
        '<a id="aieng-dash-link" href="#">Dashboard</a>' +
        "</div></div>";
      document.body.appendChild(hud);

      document.getElementById("aieng-hud-toggle").addEventListener("click", function () {
        hud.classList.toggle("collapsed");
        var ch = document.getElementById("aieng-hud-chevron");
        if (ch) ch.textContent = hud.classList.contains("collapsed") ? "▸" : "▾";
      });

      // Resolve dashboard link relative to site root
      var dash = document.getElementById("aieng-dash-link");
      if (dash) {
        var base = document.querySelector("base");
        var root = (base && base.href) || "/";
        // Prefer MkDocs path; fall back to relative guess
        var candidates = [
          "getting-started/progress/",
          "../getting-started/progress/",
          "../../getting-started/progress/",
          "getting-started/progress.html",
          "../getting-started/progress.html"
        ];
        // Try from current path depth
        var parts = location.pathname.replace(/\/+$/, "").split("/");
        // strip filename if present
        if (parts[parts.length - 1].indexOf(".") !== -1) parts.pop();
        var depth = Math.max(0, parts.length - 1);
        // Prefer absolute-from-site if repo project pages: keep relative
        dash.href = depth <= 1 ? "getting-started/progress/" : "../getting-started/progress/";
        if (location.pathname.indexOf("/core/") !== -1) {
          dash.href = "../getting-started/progress/";
        } else if (location.pathname.indexOf("/getting-started/") !== -1) {
          dash.href = "progress/";
        } else if (location.pathname.indexOf("/tracks/") !== -1) {
          dash.href = "../getting-started/progress/";
        } else if (location.pathname.indexOf("/reference/") !== -1) {
          dash.href = "../getting-started/progress/";
        } else {
          dash.href = "getting-started/progress/";
        }
      }
    }

    var lvl = levelFromXp(state.xp);
    var into = xpIntoLevel(state.xp);
    var pct = Math.min(100, Math.round((into / XP_PER_LEVEL) * 100));
    var mods = Object.keys(state.modules || {}).length;

    var elLvl = document.getElementById("aieng-level");
    var elXp = document.getElementById("aieng-xp");
    var elBar = document.getElementById("aieng-bar");
    var elMods = document.getElementById("aieng-mods");
    if (elLvl) elLvl.textContent = String(lvl);
    if (elXp) elXp.textContent = state.xp + " / next " + (lvl * XP_PER_LEVEL);
    if (elBar) elBar.style.width = pct + "%";
    if (elMods) elMods.textContent = mods + "/17";

    var badgeBox = document.getElementById("aieng-hud-badges");
    if (badgeBox) {
      var earned = BADGES.filter(function (b) {
        return state.badges[b.id];
      }).slice(-4);
      badgeBox.innerHTML = earned
        .map(function (b) {
          return '<span class="aieng-badge-chip" title="' + b.name + '">' + b.icon + " " + b.name + "</span>";
        })
        .join("");
    }
  }

  function onClickOnce(el, handler) {
    // Material instant navigation re-runs init(); avoid stacking listeners.
    if (el._aiengBound) return;
    el._aiengBound = true;
    el.addEventListener("click", handler);
  }

  function wireQuizzes(state) {
    var quizzes = document.querySelectorAll(".aieng-quiz[data-quiz-id]");
    quizzes.forEach(function (quiz) {
      var qid = quiz.getAttribute("data-quiz-id");
      var xp = parseInt(quiz.getAttribute("data-xp") || "25", 10);
      var feedback = quiz.querySelector(".quiz-feedback");
      var opts = quiz.querySelectorAll("button.quiz-opt, .quiz-opt");
      var already = !!(state.quizzes && state.quizzes[qid]);

      if (already) {
        opts.forEach(function (btn) {
          btn.disabled = true;
          if (btn.getAttribute("data-correct") === "true") btn.classList.add("correct");
        });
        if (feedback) {
          feedback.textContent = "Already solved (+" + xp + " XP saved).";
          feedback.classList.add("ok");
        }
        return;
      }

      opts.forEach(function (btn) {
        onClickOnce(btn, function () {
          if (btn.disabled) return;
          // Re-load state so multi-tab / re-init stays consistent
          state = load();
          if (state.quizzes && state.quizzes[qid]) return;

          var correct = btn.getAttribute("data-correct") === "true";
          opts.forEach(function (b) {
            b.disabled = true;
            if (b.getAttribute("data-correct") === "true") b.classList.add("correct");
          });
          if (correct) {
            btn.classList.add("correct");
            if (feedback) {
              feedback.textContent =
                (quiz.getAttribute("data-success") || "Correct.") + " (+" + xp + " XP)";
              feedback.classList.add("ok");
            }
            state.quizzes[qid] = { at: new Date().toISOString(), xp: xp };
            addXp(state, xp, "quiz");
          } else {
            btn.classList.add("wrong");
            if (feedback) {
              feedback.textContent =
                quiz.getAttribute("data-fail") ||
                "Not quite — read the explainer above and try the next quiz.";
              feedback.classList.add("bad");
            }
            // No XP on wrong; no retry this session (answers stay revealed).
            save(state);
            renderHud(state);
          }
        });
      });
    });
  }

  function wireComplete(state) {
    var boxes = document.querySelectorAll(".aieng-complete[data-module-id]");
    boxes.forEach(function (box) {
      var mid = box.getAttribute("data-module-id");
      var mod = moduleById(mid);
      var xp = parseInt(box.getAttribute("data-xp") || (mod ? mod.xp : 100), 10);
      var btn = box.querySelector("button");
      if (!btn) return;

      if (state.modules && state.modules[mid]) {
        btn.disabled = true;
        btn.textContent = "Completed ✓";
        return;
      }

      onClickOnce(btn, function () {
        state = load();
        if (state.modules[mid]) return;
        state.modules[mid] = { at: new Date().toISOString(), xp: xp };
        btn.disabled = true;
        btn.textContent = "Completed ✓";
        addXp(state, xp, "module " + mid + " complete");
      });
    });
  }

  function wireThinks(state) {
    // Small XP for revealing a "Think about it" answer once
    document.querySelectorAll(".aieng-think details[data-think-id]").forEach(function (det) {
      var tid = det.getAttribute("data-think-id");
      if (det._aiengBound) return;
      det._aiengBound = true;
      det.addEventListener("toggle", function () {
        if (!det.open) return;
        state = load();
        if (state.thinks && state.thinks[tid]) return;
        state.thinks = state.thinks || {};
        state.thinks[tid] = new Date().toISOString();
        addXp(state, 5, "reflection");
      });
    });
  }

  function renderDashboard(state) {
    var root = document.getElementById("aieng-dashboard");
    if (!root) return;

    var lvl = levelFromXp(state.xp);
    var into = xpIntoLevel(state.xp);
    var mods = Object.keys(state.modules || {}).length;
    var quizzes = Object.keys(state.quizzes || {}).length;
    var badges = Object.keys(state.badges || {}).length;

    root.innerHTML =
      '<div class="aieng-dash">' +
      '<div class="aieng-dash-grid">' +
      stat("Level", lvl) +
      stat("Total XP", state.xp) +
      stat("To next level", XP_PER_LEVEL - into) +
      stat("Modules", mods + " / 17") +
      stat("Quizzes", quizzes) +
      stat("Badges", badges + " / " + BADGES.length) +
      "</div>" +
      "<h3>Core modules</h3>" +
      '<ul class="aieng-module-list" id="aieng-mod-list"></ul>' +
      "<h3>Badges</h3>" +
      '<div class="aieng-badge-grid" id="aieng-badge-grid"></div>' +
      '<div class="aieng-reset"><button type="button" id="aieng-reset">Reset local progress</button></div>' +
      '<p class="aieng-quiz-meta" style="margin-top:0.75rem;opacity:0.8">Progress is stored only in this browser (localStorage). Clearing site data resets XP. Safe for static GitHub Pages — no accounts, no server.</p>' +
      "</div>";

    var list = document.getElementById("aieng-mod-list");
    MODULES.forEach(function (m) {
      var done = !!(state.modules && state.modules[m.id]);
      var li = document.createElement("li");
      li.innerHTML =
        '<span class="dot ' +
        (done ? "done" : "") +
        '"></span><span><strong>' +
        m.id +
        "</strong> — " +
        m.title +
        (done ? " · done" : "") +
        "</span>";
      list.appendChild(li);
    });

    var grid = document.getElementById("aieng-badge-grid");
    BADGES.forEach(function (b) {
      var earned = !!(state.badges && state.badges[b.id]);
      var card = document.createElement("div");
      card.className = "aieng-badge-card" + (earned ? " earned" : "");
      card.innerHTML =
        '<div class="icon">' +
        b.icon +
        '</div><div class="name">' +
        b.name +
        '</div><div class="desc">' +
        b.desc +
        "</div>";
      grid.appendChild(card);
    });

    var reset = document.getElementById("aieng-reset");
    if (reset) {
      reset.addEventListener("click", function () {
        if (!confirm("Reset all local XP, modules, quizzes, and badges?")) return;
        localStorage.removeItem(STORAGE_KEY);
        var fresh = defaultState();
        renderHud(fresh);
        renderDashboard(fresh);
        toast("Progress reset");
      });
    }
  }

  function stat(k, v) {
    return (
      '<div class="aieng-stat"><div class="k">' +
      k +
      '</div><div class="v">' +
      v +
      "</div></div>"
    );
  }

  function init() {
    var state = load();
    awardBadges(state, true);
    save(state);
    renderHud(state);
    wireQuizzes(state);
    wireComplete(state);
    wireThinks(state);
    renderDashboard(state);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // MkDocs instant navigation
  document.addEventListener("DOMContentLoaded", function () {
    // re-init on material instant navigation if available
  });
  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(function () {
      init();
    });
  }
})();
