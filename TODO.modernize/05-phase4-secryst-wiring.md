# Phase 4 — Wire secryst into TS + Ruby runtimes

## Goal
The new secryst ONNX is callable from both `interscript-ts` (browser)
and `interscript-ruby` (server) via the existing ML model dispatch.

## Why this phase is "just wiring"
The interpreter async path is already exercised by rababa. Adding
secryst means:
1. A new model directory in TS (`src/ml/models/secryst/`)
2. A new stdlib function (`secryst()` in `src/stdlib/ml.ts`)
3. A Ruby adapter (`Interscript::Stdlib::Functions::SecrystAdapter`)
4. An ISC test vector that exercises the wiring

## Tasks

### 4.1 TS: `src/ml/models/secryst/`

Mirror `src/ml/models/rababa/` structure:
```
secryst/
├── index.ts              # registerModel("secryst", ...)
├── diacritizer.ts        # Diacritizer class (rename to transliterator?)
├── encoder.ts            # ThaiEncoder
├── ipa.ts                # IPA constants
└── cleaner.ts            # Thai cleaner
```

Vocabulary mirrors Thai-IPA dataset (`THAI_ALPHABET`, `IPA_ALPHABET`).

Diacritizer:
```typescript
class SecrystModelImpl implements MLModel {
  readonly kind: ModelKind = "secryst"
  async transliterate(text: string): Promise<string> {
    const seq = this.encoder.inputToSequence(this.encoder.clean(text))
    const tensor = {
      name: "src", type: "int64" as const,
      data: new BigInt64Array(...),
      dims: [batchSize, maxLen],
    }
    const out = await this.session.run({ src: tensor })
    return this.decoder.decode(out)
  }
}
```

### 4.2 TS: `src/stdlib/ml.ts`

```typescript
import { createSecrystModel } from "../ml/models/secryst/index.js"

export async function secryst(input: string, opts: { config?: string } = {}): Promise<string> {
  const configKey = opts.config ?? "thai-ipa-v1"
  const model = await getSecryst(configKey)
  return model.transliterate(input)
}

const secrystConfigs = new Map<string, SecrystConfigEntry>([
  ["thai-ipa-v1", {
    model: "https://cdn.jsdelivr.net/gh/interscript/secryst@secryst_thai_ipa-v1.0.0/models/secryst_thai_ipa-v1.0.0-q8.onnx",
    config: { max_len: 256, batch_size: 16 },
  }],
])
```

Add `setSecrystConfig(key, entry)`, `resetSecrystConfigs()` mirroring rababa.

### 4.3 TS: `src/runtime/interpreter.ts`

Add secryst dispatch:
```typescript
if (rule.name === "secryst") {
  const { secryst } = await import("../stdlib/ml.js")
  ctx.current = await secryst(ctx.current, rule.kwargs as { config?: string })
  return
}
```

(`secryst` is already in `ASYNC_FUNCTIONS` set.)

### 4.4 TS: spec coverage

Add to `test/isc/parser.test.ts`:
```typescript
it("parses secryst directive", () => {
  const doc = parseIsc(minimalWrap(`
stage main {
  secryst config: "thai-ipa-v1"
}
`))
  expect(doc.stages[0]?.body[0]).toMatchObject({
    kind: "funcall", name: "secryst", kwargs: { config: "thai-ipa-v1" },
  })
})
```

### 4.5 Ruby: `Interscript::Stdlib::Functions::SecrystAdapter`

Mirror `rababa_adapter.rb`:
```ruby
class SecrystAdapter
  def call(output, config:)
    model = model_for(config)
    model.transliterate_text(output)
  end
  # ...
end
```

### 4.6 ISC grammar (TODO 10)
Already covered in `[[grammar-stages]]` (`funcall-directive`). No
change needed; secryst is just another `funcall-name` registered at
runtime.

### 4.7 End-to-end test
Add a test vector to `interscript-ml/tests/test_secryst_thai_ipa.py`:
```python
@pytest.mark.parametrize("thai,expected_ipa", [
    ("ภาษาไทย", "pʰaːsaːtʰaj"),
    ...
])
def test_parity(model_path, vocab_path, thai, expected_ipa):
    model = load_secryst(model_path, vocab_path)
    actual = model.transliterate(thai)
    assert actual == expected_ipa
```

For TS-side parity, add a test to `test/isc/end-to-end.test.ts` with
`hasAsync=true` for secryst maps.

## Acceptance
- [ ] TS `secryst()` function callable from maps.
- [ ] Ruby `SecrystAdapter` loads ONNX and runs inference.
- [ ] End-to-end test passes for ≥ 1 secryst map test vector.
- [ ] No regression in rababa path.

## Open questions
1. **Thai maps using secryst**: are there maps in `maps/` that call `secryst`? Search repo. If not, the first secryst end-to-end map is also new.
2. **Where does secryst ONNX load?** Default to CDN via the new `secrystConfigs` registry, mirror rababa's UX.
