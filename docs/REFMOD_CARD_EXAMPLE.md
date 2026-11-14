# Writing RefMod cards (H3 Scene Chain + RefMod)

The **H3 Scene Chain + RefMod** workflow reads RefMods from the asset cards in the
linked Project folder. Each card is the usual `card.md` under `characters/`,
`places/` or `props/`. To give a card a RefMod, add a `## RefMod` section. That is
the only change; every other section works exactly as before.

## Example: character card

`characters/mara/card.md`:

```markdown
# Mara
Role: orchard worker; the calm one who answers the questions (bible/mara.md).

## RefMod
character/mara_20261006_104720_aaa4dc

## Images
- `mara_sheet.png` (1672x941): FULL character sheet. Front, 3/4, side, back full-body on grey; neutral/speaking/smiling face close-ups; hands; boots. Short red hair, green cap, grey work jacket, canvas trousers. USE THIS ONE.

## Look (bible/mara.md)
- Woman, early thirties. Grey work jacket, canvas trousers, green cap. Freckles. Gloves tucked in a pocket. The cap is how you find her.

## Voice / delivery
- British English. Warm, unhurried.

## Constraints
- She answers questions; she does not narrate. Do not play her as loud or frantic.

## Verbatim constraints
Mara always wears her green cap.
```

## Example: prop card

`props/delivery_van/card.md`:

```markdown
# Delivery van

## RefMod
- `asset/delivery_van_20261006_110541_fd4358.safetensors`

## Images
- `van_exterior.png`: white panel van with a dented side door, crates of apples in the back.
- `van_interior.png`: dashboard view.

## Use
Loading and driving between scenes; do not open scenes with it.
```

## Rules for the `## RefMod` section

1. **Heading:** exactly `## RefMod` (or `## RefMods`) on its own line.
2. **Contents:** the RefMod's name as ComfyUI's **Load H3 RefMods** dropdown shows
   it: the folder under `models/refmods/`, a slash, then the file name. For example
   `character/mara_20261006_104720_aaa4dc`. One name per line.
   - A leading `- `, surrounding backticks and a trailing `.safetensors` are all
     accepted (the prop example uses them).
   - **Every non-blank line in this section is read as a name.** Put no notes or
     descriptions here; put them in another section.
   - The section ends at the next heading.
3. **Use the exported package**, the file refmodBuilder saved under
   `models/refmods/character|asset|location/`. Names starting with
   `refmodBuilder_work/` are intermediate files and are rejected.
4. **The RefMod replaces the card's pictures.** When the agent uses a card's RefMod,
   no picture from that card's folder goes into the video. Keep the `## Images`
   section anyway: the plain **H3 Scene Chain** workflow still uses it.
5. **`## Verbatim constraints` still apply.** That text is copied word for word into
   every segment where the RefMod subject is defined, exactly as for picture
   subjects.
6. **Look and Voice sections matter more.** RefMods give the model references but
   no `<Picture N>` label, so the agent names the subject in text from the card's
   look and rules.
7. **Limits:**
   - At most 8 RefMods in one video.
   - Visual only: the audio inside a package is not loaded.
   - No combined token limit for now, while the usable range is tested. Tested
     packages ranged from about 5,800 to about 19,200 tokens each.

## Checking a card before a run

Choose **H3 Scene Chain + RefMod — Local INT8 (built-in)** and run the workflow
check (preflight). The **h3scene_refmods** line shows one of:

- **pass:** every card's RefMod is installed in ComfyUI. Their names are listed.
- **pass, "No asset card names a RefMod":** no card has a `## RefMod` section yet,
  so the run uses pictures only.
- **fail:** a card names a RefMod ComfyUI doesn't have (a typo, or not copied into
  `models/refmods/`), or names a refmodBuilder work file. The message names the card.
- **unverified:** ComfyUI or the cards couldn't be read.

Optionally, add a note to `INDEX.md` such as "Mara and the delivery van have
RefMods" so the index matches the cards. The workflow does not require it.
