/*
 * MemJudge chart builders (Apache ECharts option objects).
 * Shared by the CLI HTML report and the leaderboard website so both look the same.
 * Every function is pure: data in, ECharts option out. Colours come from CSS tokens.
 */
(function (root) {
  "use strict";

  function token(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function theme() {
    return {
      text: token("--mj-text", "#1b1f24"),
      muted: token("--mj-muted", "#5b6470"),
      grid: token("--mj-grid", "#e3e6ea"),
      series: [
        token("--mj-s1", "#3b6fd8"),
        token("--mj-s2", "#d9822b"),
        token("--mj-s3", "#2f9e72"),
        token("--mj-s4", "#a24fc2"),
      ],
      good: token("--mj-good", "#2f9e72"),
      bad: token("--mj-bad", "#c94a4a"),
    };
  }

  function base(t) {
    return {
      textStyle: { color: t.text, fontFamily: "inherit" },
      color: t.series,
      animation: false,
      grid: { left: 56, right: 24, top: 40, bottom: 48, containLabel: true },
      tooltip: { confine: true },
    };
  }

  function axis(t, name, extra) {
    return Object.assign(
      {
        name: name,
        nameLocation: "middle",
        nameGap: 34,
        nameTextStyle: { color: t.muted },
        axisLine: { lineStyle: { color: t.grid } },
        axisLabel: { color: t.muted },
        splitLine: { lineStyle: { color: t.grid } },
      },
      extra || {}
    );
  }

  var pct = function (v) { return (v * 100).toFixed(1) + "%"; };

  // Tooltip formatters return HTML: run labels and category names come from result data.
  var ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) { return ESCAPES[c]; });
  }

  /** Headline: accuracy vs mean context tokens, one point per run. */
  function accuracyVsX(runs, xLabel, xOf) {
    var t = theme();
    return Object.assign(base(t), {
      tooltip: {
        formatter: function (p) {
          return esc(p.data.label) + "<br>" + xLabel + ": " + p.data.value[0].toLocaleString() +
            "<br>Accuracy: " + pct(p.data.value[1]);
        },
      },
      // Pad the range so point labels at the extremes are not clipped.
      xAxis: axis(t, xLabel, {
        type: "value",
        min: function (v) { return Math.max(0, v.min - (v.max - v.min || v.max || 1) * 0.25); },
        max: function (v) { return v.max + (v.max - v.min || v.max || 1) * 0.25; },
        axisLabel: { color: t.muted, formatter: function (v) { return +v.toFixed(2); } },
      }),
      yAxis: axis(t, "Accuracy", { type: "value", min: 0, max: 1, nameGap: 52, axisLabel: { color: t.muted, formatter: pct } }),
      series: runs.map(function (r) {
        return {
          type: "scatter",
          name: r.label,
          symbolSize: 14,
          data: [{ value: [xOf(r), r.metrics.overall.accuracy], label: r.label }],
          label: { show: true, position: "top", distance: 10, color: t.text, formatter: function (p) { return p.data.label; } },
        };
      }),
    });
  }

  /** Per-category accuracy, grouped bars per run, with 95% CI whiskers. */
  function accuracyByCategory(runs) {
    var t = theme();
    var cats = [];
    runs.forEach(function (r) {
      Object.keys(r.metrics.by_category).forEach(function (c) { if (cats.indexOf(c) < 0) cats.push(c); });
    });
    var series = [];
    runs.forEach(function (r) {
      series.push({
        type: "bar",
        name: r.label,
        data: cats.map(function (c) {
          var m = r.metrics.by_category[c];
          return m ? { value: m.accuracy, n: m.questions, ci: m.accuracy_ci95 } : null;
        }),
      });
    });
    return Object.assign(base(t), {
      legend: { show: runs.length > 1, top: 0, textStyle: { color: t.text } },
      tooltip: {
        trigger: "item",
        formatter: function (p) {
          var d = p.data;
          return esc(p.seriesName) + "<br>" + esc(p.name) + ": " + pct(d.value) + " (n=" + d.n + ")<br>95% CI " +
            pct(d.ci[0]) + "–" + pct(d.ci[1]);
        },
      },
      xAxis: axis(t, "", { type: "category", data: cats, axisLabel: { color: t.muted, interval: 0, rotate: cats.length > 5 ? 25 : 0 } }),
      yAxis: axis(t, "Accuracy", { type: "value", min: 0, max: 1, nameGap: 52, axisLabel: { color: t.muted, formatter: pct } }),
      series: series,
    });
  }

  /** Histogram of retrieval latencies (seconds) for one run. */
  function latencyHistogram(latencies) {
    var t = theme();
    if (!latencies.length) return null;
    var max = Math.max.apply(null, latencies);
    var bins = Math.min(30, Math.max(5, Math.ceil(Math.sqrt(latencies.length))));
    var width = max / bins || 1;
    var counts = new Array(bins).fill(0);
    latencies.forEach(function (v) { counts[Math.min(bins - 1, Math.floor(v / width))]++; });
    var labels = counts.map(function (_, i) { return (i * width).toFixed(2) + "–" + ((i + 1) * width).toFixed(2); });
    return Object.assign(base(t), {
      tooltip: { trigger: "axis" },
      xAxis: axis(t, "Retrieval latency (s)", { type: "category", data: labels }),
      yAxis: axis(t, "Questions", { type: "value", nameGap: 44 }),
      series: [{ type: "bar", data: counts, barCategoryGap: "8%" }],
    });
  }

  root.MemJudgeCharts = {
    accuracyVsTokens: function (runs) {
      return accuracyVsX(runs, "Mean context tokens per query", function (r) { return Math.round(r.metrics.overall.context_tokens_mean); });
    },
    accuracyVsLatency: function (runs) {
      return accuracyVsX(runs, "Retrieval latency p50 (s)", function (r) { return r.metrics.overall.retrieval_latency_p50_s || 0; });
    },
    accuracyByCategory: accuracyByCategory,
    latencyHistogram: latencyHistogram,
  };
})(window);
