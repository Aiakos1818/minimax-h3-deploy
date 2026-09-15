#!/usr/bin/env python3
"""ui2api.py -- convert a flat ComfyUI UI workflow into an API-format graph.

Only flat graphs are converted (workflows/{minimax_h3_int4clip_int8unet_raylight,
video_minimax_h3_raylight_fl2v,video_minimax_h3_raylight_ref2v}.json). The two
official templates ship as subgraphs and keep their hand-converted API files.

Rules:
  * widget values are matched to the node's widget inputs in workflow order, then
    filtered through /object_info -- that drops frontend-only widgets such as
    SaveVideo's "format.codec" sub-widget or XFuser's trailing "randomize";
  * links become [source_node_id, output_slot];
  * MarkdownNote/Note are skipped (frontend-only); other unlinked nodes are kept
    so an anchor LoadImage stays visible in the template (connect it to
    MiniMaxH3ImageToVideo.first_frame for i2v).

Usage:
  python scripts/ui2api.py workflows/video_minimax_h3_raylight_fl2v.json \
      -o workflows/api/api_video_minimax_h3_raylight_fl2v.json
"""
import argparse
import json
import sys
import urllib.request


def fetch_object_info(server):
    try:
        with urllib.request.urlopen(server + "/object_info", timeout=30) as resp:
            return json.load(resp)
    except Exception as exc:
        sys.exit("ui2api: cannot read %s/object_info (%s)\n"
                 "        start ComfyUI (scripts/start-comfyui-for-minimax-h3.sh) or pass --server" % (server, exc))


def declared_inputs(info, class_type):
    """Input names /object_info declares for a node, in INPUT_TYPES order."""
    spec = info.get(class_type, {}).get("input", {})
    names = []
    for cat in ("required", "optional"):
        names.extend((spec.get(cat) or {}).keys())
    return names, spec


def convert(ui, info):
    nodes = ui.get("nodes") or []
    links = ui.get("links") or []
    source = {}
    for link in links:
        source[(str(link[3]), link[4])] = [str(link[1]), link[2]]

    graph = {}
    for node in nodes:
        class_type = node.get("type")
        if class_type in ("MarkdownNote", "Note"):
            continue
        if class_type not in info:
            sys.exit("ui2api: node type %r is not registered in ComfyUI" % class_type)
        nid = str(node["id"])
        declared, spec = declared_inputs(info, class_type)

        widgets = node.get("widgets_values")
        widgets = widgets if isinstance(widgets, list) else ([] if widgets is None else [widgets])

        inputs = {}
        widget_index = 0
        for slot, decl in enumerate(node.get("inputs") or []):
            name = decl.get("name")
            linked = (nid, slot) in source
            if decl.get("widget"):
                if widget_index >= len(widgets):
                    sys.exit("ui2api: node %s (%s) ran out of widget values at %r" % (nid, class_type, name))
                value = widgets[widget_index]
                widget_index += 1
                # a widget converted to an input keeps its (stale) widget value in
                # the file -- the link wins, the value is only consumed for order
                if linked:
                    inputs[name] = source[(nid, slot)]
                elif name in declared:
                    inputs[name] = value
                continue
            if linked:
                inputs[name] = source[(nid, slot)]

        # Widgets the workflow file does not name (SelectVAEDevice.device, ...)
        # keep their value; map the leftovers onto the remaining declared inputs.
        link_names = {decl.get("name") for decl in (node.get("inputs") or []) if not decl.get("widget")}
        remaining = [name for name in declared if name not in inputs and name not in link_names]
        for name, value in zip(remaining, widgets[widget_index:]):
            inputs[name] = value

        # Files saved before a node gained widgets lack those values; required
        # widgets (RayInitializer.reuse_epoch, ...) take their declared default.
        for name, decl in (spec.get("required") or {}).items():
            if name in inputs or name in link_names:
                continue
            if not isinstance(decl, (list, dict)):
                continue  # a link input, not a widget
            decl_type = decl[0] if isinstance(decl, list) and decl else decl.get("type") if isinstance(decl, dict) else None
            if decl_type == "*" or (isinstance(decl_type, str) and "AUTOGROW" in decl_type.upper()):
                continue  # dynamic / autogrow group (ComfyMathExpression.values)
            if any(existing.startswith(name + ".") for existing in inputs):
                continue  # group members (values.a) are already carried by the generated names
            meta = decl[1] if isinstance(decl, list) and len(decl) > 1 else decl
            default = meta.get("default") if isinstance(meta, dict) else None
            if default is None:
                sys.exit("ui2api: node %s (%s) is missing required widget %r without a default"
                         % (nid, class_type, name))
            inputs[name] = default

        graph[nid] = {"class_type": class_type, "inputs": inputs}

    for nid, node in graph.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                if value[0] not in graph:
                    sys.exit("ui2api: node %s.%s links to missing node %s" % (nid, name, value[0]))
    return graph


def check_against_ui(graph, ui):
    """Every executable UI node must be present with the same class_type."""
    expect = {str(n["id"]): n["type"] for n in (ui.get("nodes") or [])
              if n.get("type") not in ("MarkdownNote", "Note")}
    got = {nid: node["class_type"] for nid, node in graph.items()}
    if expect != got:
        missing = {k: v for k, v in expect.items() if k not in got}
        extra = {k: v for k, v in got.items() if k not in expect}
        sys.exit("ui2api: node mismatch (missing=%s extra=%s)" % (missing, extra))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ui_json")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    args = ap.parse_args()

    with open(args.ui_json) as fh:
        ui = json.load(fh)
    if ui.get("definitions", {}).get("subgraphs"):
        sys.exit("ui2api: %s uses subgraphs; keep its hand-converted API file" % args.ui_json)

    info = fetch_object_info(args.server)
    graph = convert(ui, info)
    check_against_ui(graph, ui)

    with open(args.output, "w") as fh:
        json.dump(graph, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("%s -> %s (%d nodes)" % (args.ui_json, args.output, len(graph)))


if __name__ == "__main__":
    main()
