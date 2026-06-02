(function () {
  const colors = {
    sky: "#9beaf2",
    earth: "#a3afbf",
    edge: "rgba(78, 56, 41, 0.8)",
    bar: "#236e8c",
    barSuccess: "#5f9965",
    barFail: "#b54d3a",
    correctionBar: "rgba(255, 126, 67, 0.50)",
    correctionBarSolid: "#ff7e43",
    correctionNodeFree: "rgba(255, 179, 132, 0.69)",
    correctionNodeSupport: "rgba(255, 126, 67, 0.77)",
    correctionOutline: "rgba(255, 240, 214, 0.83)",
    ghostBar: "rgba(160, 165, 170, 0.47)",
    ghostNode: "rgba(160, 165, 170, 0.63)",
    ghostOutline: "rgba(120, 125, 130, 0.71)",
    ghostSelected: "rgba(200, 205, 210, 0.71)",
    ghostCandidateColors: [
      "rgba(160, 165, 170, 0.47)",
      "rgba(170, 150, 160, 0.47)",
      "rgba(150, 165, 175, 0.47)",
      "rgba(175, 170, 155, 0.47)",
      "rgba(155, 170, 165, 0.47)",
    ],
    freeNode: "#eee3bf",
    leftSupport: "#f2acb9",
    rightSupport: "#f2acb9",
    hover: "#ecc252",
    previewValid: "#d2f25e",
    previewInvalid: "#f26b83",
    deflectionOverlay: "rgba(224, 65, 31, 0.48)",
    text: "#f4f1eb",
  };

  function computeViewport(chasmWidth, constants) {
    const worldXmin = -constants.cliff_width - 1;
    const worldXmax = chasmWidth + constants.cliff_width + 1;
    const worldXspan = worldXmax - worldXmin;
    const scale = (constants.canvas_size - 2 * constants.render_pad) / worldXspan;
    const skyPixels = constants.horizon_frac * constants.canvas_size - constants.render_pad;
    const groundPixels = (1 - constants.horizon_frac) * constants.canvas_size - constants.render_pad;
    return {
      scale,
      world_xmin: worldXmin,
      world_xmax: worldXmax,
      world_ymin: -(groundPixels / scale),
      world_ymax: skyPixels / scale,
    };
  }

  function constantsForCanvas(canvas, constants) {
    const baseSize = Number(constants.canvas_size) || 800;
    const canvasSize = Math.max(1, Math.min(canvas.width || baseSize, canvas.height || baseSize));
    const scale = canvasSize / baseSize;
    return {
      ...constants,
      canvas_size: canvasSize,
      render_pad: constants.render_pad * scale,
      node_render_radius: constants.node_render_radius * scale,
      support_render_radius: constants.support_render_radius * scale,
      render_scale: scale,
    };
  }

  function worldToScreen(point, viewport, constants) {
    return {
      x: Math.round((point[0] - viewport.world_xmin) * viewport.scale + constants.render_pad),
      y: Math.round((viewport.world_ymax - point[1]) * viewport.scale + constants.render_pad),
    };
  }

  function screenToWorld(point, viewport, constants) {
    return [
      (point.x - constants.render_pad) / viewport.scale + viewport.world_xmin,
      viewport.world_ymax - (point.y - constants.render_pad) / viewport.scale,
    ];
  }

  function pointerToCanvas(event, canvas) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  }

  function nodeColor(node) {
    if (node.kind === "left_support") return colors.leftSupport;
    if (node.kind === "right_support") return colors.rightSupport;
    return colors.freeNode;
  }

  function isTrainingZones(state) {
    return state && state.endpoint_mode === "training_zones";
  }

  function zoneEndpoints(state, constants) {
    const startWidth = Number(constants.train_start_zone_width || constants.member_length || 3);
    const goalWidth = Number(constants.train_goal_zone_width || constants.cliff_width || 5);
    return {
      left: state.left_start_zone || [[-Math.min(constants.cliff_width, startWidth), 0], [0, 0]],
      right: state.right_goal_zone || [[state.chasm_width, 0], [state.chasm_width + Math.min(constants.cliff_width, goalWidth), 0]],
    };
  }

  function drawLine(ctx, p0, p1, color, width) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
    ctx.stroke();
  }

  function drawDotted(ctx, p0, p1, color, width) {
    ctx.save();
    ctx.setLineDash([10, 10]);
    drawLine(ctx, p0, p1, color, width);
    ctx.restore();
  }

  function barPoint(bar, key, nodesById) {
    if (bar[key]) return bar[key];
    const node = nodesById.get(Number(key === "anchor" ? bar.node_u : bar.node_v));
    return node ? [node.x, node.y] : [0, 0];
  }

  function drawBars(ctx, bars, nodesById, viewport, constants, lineWidth, colorForBar, width) {
    bars.forEach((bar) => {
      const p0 = worldToScreen(barPoint(bar, "anchor", nodesById), viewport, constants);
      const p1 = worldToScreen(barPoint(bar, "second", nodesById), viewport, constants);
      drawLine(ctx, p0, p1, colorForBar(bar), lineWidth(width || 5));
    });
  }

  function drawOverlayNodes(ctx, bars, nodesById, viewport, constants, lineWidth) {
    const highlighted = new Set();
    bars.forEach((bar) => {
      highlighted.add(Number(bar.node_u));
      highlighted.add(Number(bar.node_v));
    });
    highlighted.forEach((nodeId) => {
      const node = nodesById.get(nodeId);
      if (!node) return;
      const p = worldToScreen([node.x, node.y], viewport, constants);
      const radius = node.kind === "free" ? Math.max(4, 5 * constants.render_scale) : Math.max(5, 7 * constants.render_scale);
      ctx.fillStyle = node.kind === "free" ? colors.correctionNodeFree : colors.correctionNodeSupport;
      ctx.strokeStyle = colors.correctionOutline;
      ctx.lineWidth = lineWidth(2);
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }

  function drawGhostBars(ctx, barPlacements, viewport, constants, lineWidth, color, outlineColor, width) {
    if (!barPlacements || !barPlacements.length) return;
    ctx.save();
    barPlacements.forEach((placement) => {
      if (!placement.anchor || !placement.second) return;
      const p0 = worldToScreen(placement.anchor, viewport, constants);
      const p1 = worldToScreen(placement.second, viewport, constants);
      drawLine(ctx, p0, p1, color, lineWidth(width || 4));
      drawLine(ctx, p0, p1, outlineColor || colors.ghostOutline, lineWidth(Math.max(1, (width || 4) - 2)));
    });
    ctx.restore();
  }

  function drawGhostNodes(ctx, barPlacements, viewport, constants, lineWidth) {
    if (!barPlacements || !barPlacements.length) return;
    const seen = new Set();
    ctx.save();
    ctx.fillStyle = colors.ghostNode;
    barPlacements.forEach((placement) => {
      [placement.anchor, placement.second].forEach((point) => {
        if (!point) return;
        const p = worldToScreen(point, viewport, constants);
        const key = `${p.x}:${p.y}`;
        if (seen.has(key)) return;
        seen.add(key);
        ctx.beginPath();
        ctx.arc(p.x, p.y, Math.max(3, constants.node_render_radius), 0, Math.PI * 2);
        ctx.fill();
      });
    });
    ctx.restore();
  }

  function drawAssistSuggestions(ctx, assist, viewport, constants, lineWidth, options) {
    if (!assist || !assist.enabled || !assist.candidates || assist.candidates.length === 0) return;
    const selectedIdx = Math.max(0, Math.min(Number(assist.selected_candidate_index || 0), assist.candidates.length - 1));
    const displayMode = assist.display_mode === "all" ? "all" : "single";

    const suggesting = Boolean(options && options.suggesting);
    // Breathing pulse: ~0.7 Hz sine, mapped to alpha + glow strength.
    const phase = suggesting ? (performance.now() / 1000) * Math.PI * 2 * 0.7 : 0;
    const breath = suggesting ? 0.5 + 0.5 * Math.sin(phase) : 0;

    ctx.save();
    if (suggesting) {
      // Pulse the ghost layer's overall opacity between ~0.55 and 1.0,
      // and add a soft accent glow that swells with the same phase.
      ctx.globalAlpha = 0.55 + 0.45 * breath;
      ctx.shadowColor = `rgba(50, 180, 220, ${0.35 + 0.45 * breath})`;
      ctx.shadowBlur = (8 + 18 * breath) * constants.render_scale;
    }

    if (displayMode === "all") {
      assist.candidates.forEach((candidate, idx) => {
        if (idx === selectedIdx) return;
        drawGhostBars(
          ctx,
          candidate.bar_placements || [],
          viewport,
          constants,
          lineWidth,
          colors.ghostCandidateColors[idx % colors.ghostCandidateColors.length],
          colors.ghostOutline,
          3
        );
      });
    }

    const selected = assist.candidates[selectedIdx];
    if (selected) {
      drawGhostBars(
        ctx,
        selected.bar_placements || [],
        viewport,
        constants,
        lineWidth,
        displayMode === "all" ? colors.ghostSelected : colors.ghostBar,
        colors.ghostOutline,
        displayMode === "all" ? 5 : 4
      );
      drawGhostNodes(ctx, selected.bar_placements || [], viewport, constants, lineWidth);
    }

    ctx.restore();
  }

  function displacementForNode(femResult, nodeId) {
    if (!femResult || !femResult.node_displacements || nodeId === undefined || nodeId === null) return null;
    return femResult.node_displacements[nodeId] || femResult.node_displacements[String(nodeId)] || null;
  }

  function drawDisplacementOverlay(ctx, bars, nodesById, femResult, viewport, constants, lineWidth) {
    if (!femResult || !femResult.node_displacements) return;
    const deflectionScale = Number(constants.deflection_render_scale || constants.deflection_scale || 300);
    ctx.save();
    ctx.setLineDash([9 * constants.render_scale, 7 * constants.render_scale]);
    ctx.strokeStyle = colors.deflectionOverlay;
    ctx.lineWidth = lineWidth(3);
    ctx.lineCap = "round";

    bars.forEach((bar) => {
      const nodeU = nodesById.get(Number(bar.node_u));
      const nodeV = nodesById.get(Number(bar.node_v));
      const dispU = displacementForNode(femResult, bar.node_u);
      const dispV = displacementForNode(femResult, bar.node_v);
      if (!nodeU || !nodeV || !dispU || !dispV) return;
      const p0 = worldToScreen(
        [nodeU.x + Number(dispU[0]) * deflectionScale, nodeU.y + Number(dispU[1]) * deflectionScale],
        viewport,
        constants
      );
      const p1 = worldToScreen(
        [nodeV.x + Number(dispV[0]) * deflectionScale, nodeV.y + Number(dispV[1]) * deflectionScale],
        viewport,
        constants
      );
      drawLine(ctx, p0, p1, colors.deflectionOverlay, lineWidth(3));
    });

    ctx.setLineDash([]);
    bars.forEach((bar) => {
      [bar.node_u, bar.node_v].forEach((nodeId) => {
        const node = nodesById.get(Number(nodeId));
        const disp = displacementForNode(femResult, nodeId);
        if (!node || !disp) return;
        const p = worldToScreen(
          [node.x + Number(disp[0]) * deflectionScale, node.y + Number(disp[1]) * deflectionScale],
          viewport,
          constants
        );
        ctx.beginPath();
        ctx.arc(p.x, p.y, Math.max(2, 4 * constants.render_scale), 0, Math.PI * 2);
        ctx.fillStyle = colors.deflectionOverlay;
        ctx.fill();
      });
    });
    ctx.restore();
  }

  function drawScene(canvas, state, constants, options) {
    const ctx = canvas.getContext("2d");
    const chasmWidth = state ? state.chasm_width : 10;
    const renderConstants = constantsForCanvas(canvas, constants);
    const viewport = computeViewport(chasmWidth, renderConstants);
    const lineWidth = (value) => Math.max(1, value * renderConstants.render_scale);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = colors.sky;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const horizon = worldToScreen([0, 0], viewport, renderConstants).y;
    const leftEdge = worldToScreen([0, 0], viewport, renderConstants).x;
    const rightEdge = worldToScreen([chasmWidth, 0], viewport, renderConstants).x;
    ctx.fillStyle = colors.earth;
    ctx.fillRect(0, horizon, leftEdge, canvas.height - horizon);
    ctx.fillRect(rightEdge, horizon, canvas.width - rightEdge, canvas.height - horizon);
    drawLine(ctx, { x: leftEdge, y: horizon }, { x: leftEdge, y: canvas.height }, colors.edge, lineWidth(3));
    drawLine(ctx, { x: rightEdge, y: horizon }, { x: rightEdge, y: canvas.height }, colors.edge, lineWidth(3));
    drawLine(ctx, { x: 0, y: horizon }, { x: leftEdge, y: horizon }, colors.edge, lineWidth(2));
    drawLine(ctx, { x: rightEdge, y: horizon }, { x: canvas.width, y: horizon }, colors.edge, lineWidth(2));

    const trainingZones = isTrainingZones(state);
    if (trainingZones) {
      const zones = zoneEndpoints(state, renderConstants);
      drawLine(
        ctx,
        worldToScreen(zones.left[0], viewport, renderConstants),
        worldToScreen(zones.left[1], viewport, renderConstants),
        colors.leftSupport,
        lineWidth(4)
      );
      drawLine(
        ctx,
        worldToScreen(zones.right[0], viewport, renderConstants),
        worldToScreen(zones.right[1], viewport, renderConstants),
        colors.rightSupport,
        lineWidth(4)
      );
    }

    const labelSize = Math.max(8, 16 * renderConstants.render_scale);
    ctx.font = `${labelSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.fillStyle = colors.leftSupport;
    if (trainingZones) {
      const zones = zoneEndpoints(state, renderConstants);
      const startAnchor = worldToScreen([(zones.left[0][0] + zones.left[1][0]) * 0.5, 0], viewport, renderConstants);
      const endAnchor = worldToScreen([(zones.right[0][0] + zones.right[1][0]) * 0.5, 0], viewport, renderConstants);
      ctx.fillText("START ZONE", startAnchor.x - 50 * renderConstants.render_scale, startAnchor.y - 20 * renderConstants.render_scale);
      ctx.fillStyle = colors.rightSupport;
      ctx.fillText("GOAL BAND", endAnchor.x - 42 * renderConstants.render_scale, endAnchor.y - 20 * renderConstants.render_scale);
    } else {
      ctx.fillText("START", leftEdge - 24 * renderConstants.render_scale, horizon - 20 * renderConstants.render_scale);
      ctx.fillStyle = colors.rightSupport;
      ctx.fillText("END", rightEdge - 14 * renderConstants.render_scale, horizon - 20 * renderConstants.render_scale);
    }

    const bars = options && options.bars ? options.bars : state ? state.bars : [];
    const nodes = state ? state.nodes : [];
    const nodesById = new Map(nodes.map((node) => [Number(node.id), node]));
    drawBars(ctx, bars, nodesById, viewport, renderConstants, lineWidth, (bar) => {
      let color = colors.bar;
      const utilization = state && state.fem_result && state.fem_result.member_utilization
        ? state.fem_result.member_utilization[bar.id] || state.fem_result.member_utilization[String(bar.id)] || 0
        : 0;
      if (state && state.finalized && state.fem_result && Number(utilization) > 1) color = colors.barFail;
      else if (state && state.finalized && state.fem_result && state.fem_result.status === "ok") color = colors.barSuccess;
      return color;
    }, 5);

    if (options && Number.isFinite(Number(options.highlightSuffixStartBarCount))) {
      const highlightBars = bars.slice(Math.max(0, Number(options.highlightSuffixStartBarCount)));
      drawBars(ctx, highlightBars, nodesById, viewport, renderConstants, lineWidth, () => colors.correctionBarSolid, 6);
    }

    if (state && state.finalized && state.fem_result) {
      drawDisplacementOverlay(ctx, bars, nodesById, state.fem_result, viewport, renderConstants, lineWidth);
    }

    if (options && options.preview) {
      const previewSuggesting = Boolean(options.suggesting);
      const phase = previewSuggesting ? (performance.now() / 1000) * Math.PI * 2 * 0.7 : 0;
      const breath = previewSuggesting ? 0.5 + 0.5 * Math.sin(phase) : 0;
      const p0 = worldToScreen(options.preview.anchor, viewport, renderConstants);
      const p1 = worldToScreen(options.preview.second, viewport, renderConstants);
      ctx.save();
      if (previewSuggesting) {
        ctx.globalAlpha = 0.62 + 0.38 * breath;
        ctx.shadowColor = `rgba(50, 180, 220, ${0.34 + 0.46 * breath})`;
        ctx.shadowBlur = (8 + 18 * breath) * renderConstants.render_scale;
      }
      drawDotted(ctx, p0, p1, options.preview.valid === false ? colors.previewInvalid : colors.previewValid, lineWidth(4));
      ctx.fillStyle = options.preview.valid === false ? colors.previewInvalid : colors.previewValid;
      [p0, p1].forEach((point) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, Math.max(3, 5 * renderConstants.render_scale), 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.restore();
    }

    if (options && options.assist) {
      drawAssistSuggestions(ctx, options.assist, viewport, renderConstants, lineWidth, {
        suggesting: Boolean(options.suggesting),
      });
    }

    nodes.forEach((node) => {
      if (
        trainingZones
        && (node.kind === "left_support" || node.kind === "right_support")
        && (!node.incident_bar_ids || node.incident_bar_ids.length === 0)
        && node.id !== (options && options.hoveredNodeId)
        && node.id !== (options && options.selectedNodeId)
      ) {
        return;
      }
      const p = worldToScreen([node.x, node.y], viewport, renderConstants);
      const radius = node.kind === "free" ? renderConstants.node_render_radius : renderConstants.support_render_radius;
      ctx.fillStyle = nodeColor(node);
      ctx.strokeStyle = options && options.hoveredNodeId === node.id ? colors.hover : colors.edge;
      ctx.lineWidth = lineWidth(2);
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });

    if (options && options.overlay && options.overlay.state) {
      const overlayState = options.overlay.state;
      const overlayBars = (overlayState.bars || []).slice(Math.max(0, Number(options.overlay.suffixStartBarCount || 0)));
      const overlayNodes = new Map((overlayState.nodes || []).map((node) => [Number(node.id), node]));
      ctx.save();
      drawBars(ctx, overlayBars, overlayNodes, viewport, renderConstants, lineWidth, () => colors.correctionBar, 5);
      drawOverlayNodes(ctx, overlayBars, overlayNodes, viewport, renderConstants, lineWidth);
      ctx.restore();
    }

    return viewport;
  }

  function distance(a, b) {
    return Math.hypot(a[0] - b[0], a[1] - b[1]);
  }

  function resolveAnchor(world, state, constants) {
    if (!state) return null;
    let best = null;
    state.nodes.forEach((node) => {
      const d = distance(world, [node.x, node.y]);
      if (d <= constants.node_snap_radius && (!best || d < best.distance)) {
        best = { point: [node.x, node.y], nodeId: node.id, kind: node.kind, distance: d, movable: node.movable };
      }
    });
    [
      { point: [0, 0], kind: "left_support" },
      { point: [state.chasm_width, 0], kind: "right_support" },
    ].forEach((support) => {
      const d = distance(world, support.point);
      if (d <= constants.support_snap_radius && (!best || d < best.distance)) {
        best = { point: support.point, nodeId: null, kind: support.kind, distance: d, movable: false };
      }
    });
    return best;
  }

  function pickMovableNode(world, state, constants) {
    if (!state) return null;
    let best = null;
    state.nodes.forEach((node) => {
      if (!node.movable) return;
      const d = distance(world, [node.x, node.y]);
      if (d <= constants.node_snap_radius && (!best || d < best.distance)) {
        best = { node, distance: d };
      }
    });
    return best ? best.node : null;
  }

  window.BridgeRenderer = {
    computeViewport,
    worldToScreen,
    screenToWorld,
    pointerToCanvas,
    drawScene,
    resolveAnchor,
    pickMovableNode,
  };
})();
