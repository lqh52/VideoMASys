"""Request-local schema constraints; canonical evidence IDs are never rewritten."""

from copy import deepcopy

from pydantic import BaseModel, ConfigDict, Field


class ChoiceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    selected_choice: str | None = Field(max_length=300)


def bounded_schema(
    model, *, frames=None, observations=None, needs=None, options=None, targets=None,
):
    """Constrain existing fields to this request's actual registry, not global IDs.

    This narrows the wire schema only. Runtime validation still checks chronology,
    provenance, dataflow and support. An empty registry permits only empty lists.
    """
    schema = deepcopy(model.model_json_schema())
    lists = {"frame_ids": frames, "observation_ids": observations, "need_ids": needs}
    scalars = {"need_id": needs, "refine_frame_id": frames,
               "choice": options, "selected_choice": options, "target_id": targets}

    def scalar(node, allowed):
        if node.get("type") == "string":
            node["enum"] = list(allowed)
        for branch in node.get("anyOf", []):
            scalar(branch, allowed)

    def visit(node):
        if not isinstance(node, dict):
            return
        for name, field in node.get("properties", {}).items():
            if name in {"option_comparisons", "option_assessments"} and options == []:
                field["maxItems"] = 0
            if name in lists and lists[name] is not None:
                allowed = list(lists[name])
                if allowed:
                    # Identifiers may be inline or a shared $ref. A typed inline
                    # enum avoids changing other reference categories via $defs.
                    field["items"] = {"type": "string", "enum": allowed}
                else:
                    field["maxItems"] = 0
            if name in scalars and scalars[name] is not None:
                allowed = list(scalars[name])
                if allowed:
                    scalar(field, allowed)
                elif name == "selected_choice":
                    field.clear()
                    field["type"] = "null"
            visit(field)
        for definition in node.get("$defs", {}).values():
            visit(definition)
        visit(node.get("items"))
        for branch in node.get("anyOf", []):
            visit(branch)

    visit(schema)
    return schema
