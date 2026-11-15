(function () {
  "use strict";

  const notes = Array.isArray(window.LLM_NOTES) ? window.LLM_NOTES.slice() : [];
  const byId = new Map(notes.map((note) => [note.id, note]));
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");
  const THEME_KEY = "llmnote-theme";

  const phases = [
    { title: "Foundations", first: 1, last: 3 },
    { title: "Core LangChain", first: 4, last: 6 },
    { title: "Advanced Patterns", first: 7, last: 9 },
    { title: "Retrieval & RAG", first: 10, last: 11 },
    { title: "Agents & Reliability", first: 12, last: 14 },
  ];

  const els = {
    nav: document.getElementById("note-nav"),
    content: document.getElementById("content"),
    crumb: document.getElementById("crumb-title"),
    articleTime: document.getElementById("article-time"),
    topLevel: document.getElementById("top-level"),
    activeFile: document.getElementById("active-file"),
    summary: document.getElementById("notes-summary"),
    count: document.getElementById("note-count"),
    main: document.getElementById("main"),
    minimap: document.getElementById("minimap"),
    sidebar: document.getElementById("sidebar"),
    scrim: document.getElementById("scrim"),
    menuBtn: document.getElementById("menu-btn"),
    closeBtn: document.getElementById("sidebar-close"),
    brandHome: document.getElementById("brand-home"),
    toTop: document.getElementById("to-top"),
    themeToggle: document.getElementById("theme-toggle"),
    themeMeta: document.querySelector('meta[name="theme-color"]'),
  };

  const storedTheme = () => {
    try {
      return localStorage.getItem(THEME_KEY);
    } catch (error) {
      return null;
    }
  };

  const activeTheme = () =>
    document.documentElement.dataset.theme === "dark" ? "dark" : "light";

  const applyTheme = (theme) => {
    const dark = theme === "dark";
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;

    if (els.themeMeta) {
      els.themeMeta.setAttribute("content", dark ? "#121310" : "#FAF9F6");
    }

    if (els.themeToggle) {
      const label = dark ? "Switch to light theme" : "Switch to dark theme";
      els.themeToggle.setAttribute("aria-label", label);
      els.themeToggle.setAttribute("title", label);
    }
  };

  applyTheme(activeTheme());

  if (els.themeToggle) {
    els.themeToggle.addEventListener("click", () => {
      const next = activeTheme() === "dark" ? "light" : "dark";
      applyTheme(next);
      try {
        localStorage.setItem(THEME_KEY, next);
      } catch (error) {
        /* the theme still applies for this session */
      }
    });
  }

  const followSystemTheme = (event) => {
    if (!storedTheme()) {
      applyTheme(event.matches ? "dark" : "light");
    }
  };

  if (typeof systemTheme.addEventListener === "function") {
    systemTheme.addEventListener("change", followSystemTheme);
  } else {
    systemTheme.addListener(followSystemTheme);
  }

  const escapeHtml = (value) =>
    String(value).replace(/[&<>"']/g, (character) => {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[character];
    });

  const readingMinutes = (markdown) => {
    const plain = markdown
      .replace(/```[\s\S]*?```/g, " ")
      .replace(/`[^`]*`/g, " ")
      .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
      .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
      .replace(/[#>*_|~-]/g, " ");
    const words = plain.trim().split(/\s+/).filter(Boolean).length;
    return Math.max(1, Math.round(words / 220));
  };

  notes.forEach((note) => {
    note.minutes = readingMinutes(note.content);
  });

  const totalMinutes = notes.reduce((total, note) => total + note.minutes, 0);

  const markedHtml = (markdown) => {
    if (window.marked && typeof window.marked.parse === "function") {
      try {
        return window.marked.parse(markdown, { gfm: true, breaks: false });
      } catch (error) {
        /* fall back to plain text below */
      }
    }
    return `<pre><code>${escapeHtml(markdown)}</code></pre>`;
  };

  const addHeadingIds = (root) => {
    const used = new Map();

    root.querySelectorAll("h1, h2, h3, h4").forEach((heading) => {
      const base =
        heading.textContent
          .trim()
          .toLowerCase()
          .replace(/`/g, "")
          .replace(/&/g, " and ")
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/^-+|-+$/g, "") || "section";

      const count = (used.get(base) || 0) + 1;
      used.set(base, count);
      heading.id = count === 1 ? base : `${base}-${count}`;
    });
  };

  const wrapTables = (root) => {
    root.querySelectorAll("table").forEach((table) => {
      const wrapper = document.createElement("div");
      wrapper.className = "table-wrap";
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    });
  };

  const prepareLinks = (root) => {
    root.querySelectorAll("a[href]").forEach((link) => {
      const href = link.getAttribute("href") || "";

      if (/^https?:\/\//i.test(href)) {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        return;
      }

      const match = href.match(/^(?:\.\/)?([^/?#]+\.md)(?:[?#].*)?$/i);
      if (!match) return;

      const target = notes.find(
        (note) => note.file.toLowerCase() === match[1].toLowerCase()
      );

      if (target) {
        link.setAttribute("href", `#${target.id}`);
        link.removeAttribute("target");
        link.removeAttribute("rel");
      }
    });
  };

  const setMinimapActive = () => {
    if (!els.minimap) return;

    const marks = Array.from(els.minimap.querySelectorAll(".minimap-mark"));
    if (!marks.length) return;

    const threshold = els.main.getBoundingClientRect().top + 96;
    let activeId = marks[0].dataset.target;

    marks.forEach((mark) => {
      const heading = document.getElementById(mark.dataset.target);
      if (heading && heading.getBoundingClientRect().top <= threshold) {
        activeId = mark.dataset.target;
      }
    });

    marks.forEach((mark) => {
      if (mark.dataset.target === activeId) {
        mark.setAttribute("aria-current", "location");
      } else {
        mark.removeAttribute("aria-current");
      }
    });
  };

  const buildMinimap = () => {
    if (!els.minimap) return;
    els.minimap.innerHTML = "";

    els.content.querySelectorAll("h2").forEach((heading) => {
      const label = heading.textContent.trim();
      const mark = document.createElement("button");
      mark.type = "button";
      mark.className = "minimap-mark";
      mark.dataset.target = heading.id;
      mark.dataset.label = label;
      mark.setAttribute("aria-label", `Go to ${label}`);

      const bar = document.createElement("span");
      bar.className = "minimap-bar";
      mark.appendChild(bar);

      mark.addEventListener("click", () => {
        heading.scrollIntoView({
          behavior: reduceMotion ? "auto" : "smooth",
          block: "start",
        });
      });

      els.minimap.appendChild(mark);
    });

    setMinimapActive();
  };

  let minimapFrame = 0;

  const queueMinimapUpdate = () => {
    if (minimapFrame) return;
    minimapFrame = requestAnimationFrame(() => {
      minimapFrame = 0;
      setMinimapActive();
    });
  };

  const navButton = (note) => `
    <button
      type="button"
      class="nav-note flex w-full items-start gap-2.5 rounded-md px-2.5 py-1.5 text-left"
      data-note-id="${escapeHtml(note.id)}"
      aria-label="${escapeHtml(note.title)}"
    >
      <span class="nav-num mt-px w-[18px] flex-none font-mono text-[10px] font-medium leading-none text-faint">${String(note.part).padStart(2, "0")}</span>
      <span class="min-w-0 flex-1 text-[13px] leading-[1.35]">${escapeHtml(note.name)}</span>
      <span class="nav-time mt-px flex-none font-mono text-[9.5px] leading-none text-faint">${note.minutes}m</span>
    </button>
  `;

  const buildNav = () => {
    els.count.textContent = String(notes.length).padStart(2, "0");

    const overview = notes.find((note) => note.part === 0);
    const groups = [];

    if (overview) {
      groups.push(`<div class="nav-phase">${navButton(overview)}</div>`);
    }

    phases.forEach((phase, index) => {
      const items = notes.filter(
        (note) => note.part >= phase.first && note.part <= phase.last
      );
      if (!items.length) return;

      groups.push(`
        <div class="nav-phase">
          <p class="nav-phase-label">
            <span class="nav-phase-num">Phase ${String(index + 1).padStart(2, "0")}</span>
            <span class="nav-phase-name">${escapeHtml(phase.title)}</span>
            <span class="nav-phase-rule" aria-hidden="true"></span>
          </p>
          ${items.map(navButton).join("")}
        </div>
      `);
    });

    els.nav.innerHTML = groups.join("");
  };

  const setActive = (note) => {
    els.crumb.textContent = note.title;
    els.crumb.title = note.title;
    els.articleTime.textContent = `${note.minutes} min read`;
    els.topLevel.textContent =
      note.part === 0 ? "Overview" : `Part ${String(note.part).padStart(2, "0")}`;
    els.activeFile.textContent = note.file;
    els.activeFile.title = note.file;
    document.title = `${note.title} · llmnote`;

    els.nav.querySelectorAll(".nav-note").forEach((button) => {
      if (button.dataset.noteId === note.id) {
        button.setAttribute("aria-current", "page");
      } else {
        button.removeAttribute("aria-current");
      }
    });
  };

  const render = (note) => {
    els.content.innerHTML = markedHtml(note.content);
    addHeadingIds(els.content);
    wrapTables(els.content);
    prepareLinks(els.content);
    buildMinimap();

    if (window.hljs) {
      els.content.querySelectorAll("pre code").forEach((code) => {
        try {
          window.hljs.highlightElement(code);
        } catch (error) {
          /* keep the code readable if highlighting fails */
        }
      });
    }

    setActive(note);
    els.main.scrollTop = 0;
  };

  const hashId = () => {
    const raw = location.hash.replace(/^#\/?/, "");
    try {
      return decodeURIComponent(raw);
    } catch (error) {
      return raw;
    }
  };

  const route = () => {
    if (!notes.length) return;

    const note = byId.get(hashId()) || notes[0];
    if (location.hash !== `#${note.id}`) {
      history.replaceState(null, "", `#${note.id}`);
    }

    render(note);
    closeDrawer();
  };

  const drawerOpen = () => els.menuBtn.getAttribute("aria-expanded") === "true";

  function openDrawer() {
    els.sidebar.classList.remove("-translate-x-full");
    els.scrim.classList.remove("hidden");
    els.menuBtn.setAttribute("aria-expanded", "true");
  }

  function closeDrawer() {
    els.sidebar.classList.add("-translate-x-full");
    els.scrim.classList.add("hidden");
    els.menuBtn.setAttribute("aria-expanded", "false");
  }

  if (!notes.length) {
    els.crumb.textContent = "No notes";
    els.content.innerHTML =
      '<p>No notes found. Add <code>01-*.md</code> files and run <code>node build-notes.mjs</code>.</p>';
    return;
  }

  buildNav();
  els.summary.textContent = `${notes.length} notes · ${phases.length} phases · ${totalMinutes} min`;
  els.main.addEventListener("scroll", queueMinimapUpdate, { passive: true });

  els.nav.addEventListener("click", (event) => {
    const button = event.target.closest(".nav-note");
    if (!button) return;

    const id = button.dataset.noteId;
    if (!byId.has(id)) return;

    if (location.hash === `#${id}`) {
      closeDrawer();
      return;
    }

    location.hash = id;
  });

  els.brandHome.addEventListener("click", (event) => {
    event.preventDefault();
    const first = notes[0];

    if (location.hash === `#${first.id}`) {
      closeDrawer();
      return;
    }

    location.hash = first.id;
  });

  els.menuBtn.addEventListener("click", () => {
    if (drawerOpen()) {
      closeDrawer();
    } else {
      openDrawer();
    }
  });

  els.closeBtn.addEventListener("click", closeDrawer);
  els.scrim.addEventListener("click", closeDrawer);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && drawerOpen()) {
      closeDrawer();
      els.menuBtn.focus();
    }
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth >= 1024 && drawerOpen()) {
      closeDrawer();
    }
    queueMinimapUpdate();
  });

  els.toTop.addEventListener("click", () => {
    els.main.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
  });

  window.addEventListener("hashchange", route);
  route();
})();
