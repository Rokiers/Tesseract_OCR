"""
从 CSV 文件中提取游戏词汇表，输出为干净的词表文件。
"""
import csv
import json
import os
import random

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. 提取物品名称
# ============================================================
item_names = set()
with open(os.path.join(PROJECT_ROOT, 'all_item_names_en.csv'), 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row['name_en'].strip()
        if name:
            item_names.add(name)

print(f"[1] 物品名称: {len(item_names)} 个")

# ============================================================
# 2. 提取属性描述
# ============================================================
property_descs = set()
with open(os.path.join(PROJECT_ROOT, 'all_property_defs_en.csv'), 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        desc = row['desc_en'].strip()
        if desc:
            property_descs.add(desc)

print(f"[2] 属性描述: {len(property_descs)} 个")

# ============================================================
# 3. 从属性描述中提取常见的数值模板
#    e.g. "Fire Resist" -> "+X% Fire Resist", "+X Fire Resist"
# ============================================================
property_templates = set()
for desc in property_descs:
    # 给每个属性生成带数值前缀的变体
    prefix_patterns = [
        "+X% $DESC",
        "+X $DESC",
        "-X% $DESC",
        "-X $DESC",
        "$DESC +X",
        "$DESC +X%",
        "+X-$Y $DESC",
        "$DESC +X-$Y",
    ]
    for tmpl in prefix_patterns:
        property_templates.add(tmpl.replace("$DESC", desc))

    # 如果描述中有 "to"，生成 "X to Y DESC" 的模式
    property_templates.add(f"+X {desc}")
    property_templates.add(f"+X% {desc}")
    property_templates.add(f"X% {desc}")

print(f"[3] 属性模板: {len(property_templates)} 个")

# ============================================================
# 4. 固定游戏UI文本
# ============================================================
ui_texts = [
    "ONE-HAND DAMAGE",
    "TWO-HAND DAMAGE",
    "THROWING DAMAGE",
    "DURABILITY",
    "REQUIRED DEXTERITY",
    "REQUIRED STRENGTH",
    "REQUIRED LEVEL",
    "SWORD CLASS",
    "AXE CLASS",
    "BOW CLASS",
    "MACE CLASS",
    "SPEAR CLASS",
    "POLEARM CLASS",
    "JAVELIN CLASS",
    "DAGGER CLASS",
    "SCEPTER CLASS",
    "STAFF CLASS",
    "CROSSBOW CLASS",
    "SORCERESS CLASS",
    "AMAZON CLASS",
    "WAND CLASS",
    "CLAW CLASS",
    "THROWING CLASS",
    "VERY FAST ATTACK SPEED",
    "FAST ATTACK SPEED",
    "NORMAL ATTACK SPEED",
    "SLOW ATTACK SPEED",
    "VERY SLOW ATTACK SPEED",
    "SOCKETED",
    "CTRL",
    "LEFT CLICK TO MOVE",
    "HOLD SHIFT TO COMPARE",
    "INDESTRUCTIBLE",
    "REPLENISHES QUANTITY",
    "ONE HAND",
    "TWO HAND",
    "QUANTITY",
    "DAMAGE",
    "DEFENSE",
]
ui_texts_set = set(ui_texts)

# 生成 UI 文本的数值变体
ui_templates = set()
for text in ui_texts:
    ui_templates.add(text)
    # 数值化的版本：DURABILITY: X OF Y
    if "DURABILITY" in text:
        ui_templates.add("DURABILITY: X OF Y")
    if "DAMAGE" in text and ":" not in text:
        ui_templates.add(f"{text}: X TO Y")
    if "SOCKETED" in text:
        ui_templates.add("SOCKETED (X)")
    if "QUANTITY" in text:
        ui_templates.add("QUANTITY: X OF Y")

print(f"[4] UI文本: {len(ui_templates)} 个")

# ============================================================
# 4.5. 生成带数字的训练样本
# ============================================================
SMALL_NUMS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 27, 28, 30, 33, 35, 36, 39]
MED_NUMS = [40, 45, 48, 50, 55, 60, 63, 65, 68, 70, 75, 80, 85, 87, 90, 95, 99]
LARGE_NUMS = [100, 105, 110, 120, 122, 136, 146, 150, 160, 166, 172, 184, 187, 200, 208, 210, 250, 270, 300, 350, 400, 500]
PERCENT_NUMS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 33, 35, 40, 45, 50, 60, 70, 75, 80, 87, 90, 95, 99, 100, 105, 120, 150, 160, 166, 187, 200, 208, 250, 300]

ALL_NUMS = SMALL_NUMS + MED_NUMS + LARGE_NUMS
numeric_words = set()

# --- 5.1 纯数字 / +N / N% / +N% ---
for n in ALL_NUMS:
    numeric_words.add(str(n))
for p in PERCENT_NUMS:
    numeric_words.add(f"+{p}%")
    numeric_words.add(f"{p}%")
for n in SMALL_NUMS + MED_NUMS[:10]:
    numeric_words.add(f"+{n}")

# --- 5.2 范围: N TO M / N OF M / N-M ---
for _ in range(100):
    a = random.choice(ALL_NUMS)
    b = random.choice(ALL_NUMS)
    if a > b:
        a, b = b, a
    numeric_words.add(f"{a} TO {b}")
for _ in range(50):
    a = random.choice(ALL_NUMS)
    b = random.choice(ALL_NUMS)
    if a > b:
        a, b = b, a
    numeric_words.add(f"{a} OF {b}")
for _ in range(40):
    a = random.choice(ALL_NUMS)
    b = random.choice(ALL_NUMS)
    if a > b:
        a, b = b, a
    numeric_words.add(f"{a}-{b}")

# --- 5.3 属性 + 百分比前缀: +N% DESC ---
pct_props = [
    'FIRE RESIST', 'COLD RESIST', 'LIGHTNING RESIST', 'POISON RESIST',
    'ENHANCED DAMAGE', 'INCREASED ATTACK SPEED', 'FASTER CAST RATE',
    'FASTER HIT RECOVERY', 'FASTER RUN/WALK', 'FASTER BLOCK RATE',
    'CHANCE OF CRUSHING BLOW', 'DEADLY STRIKE', 'CHANCE OF OPEN WOUNDS',
    'BETTER CHANCE OF GETTING MAGIC ITEMS', 'EXTRA GOLD FROM MONSTERS',
    'LIFE STOLEN PER HIT', 'MANA STOLEN PER HIT',
    'REDUCES ALL VENDOR PRICES', 'BONUS TO ATTACK RATING',
]
for desc in pct_props:
    for _ in range(3):
        pct = random.choice(PERCENT_NUMS)
        numeric_words.add(f"+{pct}% {desc}")

# --- 5.4 属性 + 数值前缀: +N TO DESC ---
num_props = [
    'ATTACK RATING', 'STRENGTH', 'DEXTERITY', 'VITALITY', 'ENERGY',
    'LIGHT RADIUS', 'MAXIMUM DAMAGE', 'MANA', 'ALL SKILLS',
    'AMAZON SKILL LEVELS', 'BARBARIAN SKILL LEVELS',
    'SORCERESS SKILL LEVELS', 'NECROMANCER SKILL LEVELS',
    'PALADIN SKILL LEVELS', 'ASSASSIN SKILL LEVELS',
    'DRUID SKILL LEVELS',
]
for desc in num_props:
    for _ in range(3):
        n = random.choice(ALL_NUMS)
        numeric_words.add(f"+{n} TO {desc}")

# --- 5.5 属性 + 数值后缀: DESC +N / DESC +N% ---
suffix_props = ['ALL RESISTANCES', 'COLD RESIST', 'FIRE RESIST', 'LIGHTNING RESIST', 'POISON RESIST']
for desc in suffix_props:
    for _ in range(2):
        n = random.choice(ALL_NUMS[:20])
        numeric_words.add(f"{desc} +{n}")
        pct = random.choice(PERCENT_NUMS[:15])
        numeric_words.add(f"{desc} +{pct}%")

# --- 5.6 UI 模板 + 数字: SOCKETED (N) / REQUIRED X: N ---
for _ in range(20):
    n = random.choice(range(2, 7))
    numeric_words.add(f"SOCKETED ({n})")
for _ in range(20):
    n = random.choice(ALL_NUMS[:30])
    numeric_words.add(f"REQUIRED LEVEL: {n}")
    numeric_words.add(f"REQUIRED STRENGTH: {n}")
    numeric_words.add(f"REQUIRED DEXTERITY: {n}")
for _ in range(15):
    a = random.choice(ALL_NUMS[:25])
    b = random.choice(ALL_NUMS[:25])
    if a > b:
        a, b = b, a
    numeric_words.add(f"DURABILITY: {a} OF {b}")
    numeric_words.add(f"QUANTITY: {a} OF {b}")

# --- 5.7 伤害 UI 模板: ONE-HAND DAMAGE: N TO M ---
for _ in range(30):
    a = random.choice(ALL_NUMS[:20])
    b = random.choice(ALL_NUMS)
    if a > b:
        a, b = b, a
    numeric_words.add(f"ONE-HAND DAMAGE: {a} TO {b}")
    numeric_words.add(f"TWO-HAND DAMAGE: {a} TO {b}")

# --- 5.8 ADDS X-Y DAMAGE / ADDS X-Y ELEM DAMAGE ---
elements = ['COLD', 'FIRE', 'LIGHTNING', 'POISON', 'MAGIC']
for _ in range(40):
    a = random.choice(ALL_NUMS[:15])
    b = random.choice(ALL_NUMS[:25])
    if a > b:
        a, b = b, a
    if random.random() < 0.5:
        elem = random.choice(elements)
        numeric_words.add(f"ADDS {a}-{b} {elem} DAMAGE")
    else:
        numeric_words.add(f"ADDS {a}-{b} DAMAGE")

# --- 5.9 LEVEL N / N% CHANCE TO CAST LEVEL N ON STRIKING ---
for _ in range(15):
    n = random.choice(SMALL_NUMS[:15])
    numeric_words.add(f"LEVEL {n}")
for _ in range(15):
    pct = random.choice(PERCENT_NUMS[:10])
    lvl = random.choice(SMALL_NUMS[:10])
    numeric_words.add(f"{pct}% CHANCE TO CAST LEVEL {lvl} ON STRIKING")
    numeric_words.add(f"{pct}% CHANCE TO CAST LEVEL {lvl} ON ATTACK")

# --- 5.10 其他游戏常见组合 ---
combo_templates = [
    "+{N}% CHANCE OF OPEN WOUNDS",
    "+{N}% DEADLY STRIKE",
    "+{N}% ENHANCED DAMAGE",
    "+{N}% INCREASED ATTACK SPEED",
    "+{N} TO STRENGTH",
    "+{N} TO DEXTERITY",
    "+{N} TO ATTACK RATING",
    "{N}% LIFE STOLEN PER HIT",
    "{N}% MANA STOLEN PER HIT",
    "ALL RESISTANCES +{N}",
    "ALL RESISTANCES +{N}%",
]
for tmpl in combo_templates:
    for _ in range(3):
        n = random.choice(ALL_NUMS[:20])
        numeric_words.add(tmpl.replace("{N}", str(n)))

# --- 5.11 吸血/吸蓝 ---
for _ in range(10):
    pct = random.choice(PERCENT_NUMS[:12])
    numeric_words.add(f"{pct}% LIFE STOLEN PER HIT")
    numeric_words.add(f"{pct}% MANA STOLEN PER HIT")

# --- 5.12 特定暗黑属性组合 ---
game_specific = [
    "+33% CHANCE OF OPEN WOUNDS",
    "+30% DEADLY STRIKE",
    "+63% ENHANCED DAMAGE",
    "+208% ENHANCED DAMAGE",
    "+30% INCREASED ATTACK SPEED",
    "+50% INCREASED ATTACK SPEED",
    "+20% INCREASED ATTACK SPEED",
    "ADDS 30-50 DAMAGE",
    "ADDS 1-15 LIGHTNING DAMAGE",
    "ADDS 60-120 MAGIC DAMAGE",
    "ADDS 1-200 LIGHTNING DAMAGE",
    "ALL RESISTANCES +10",
    "+60 TO ATTACK RATING",
    "+5 TO STRENGTH",
    "+3 TO DEXTERITY",
    "+35 TO MANA",
    "+5 TO ENERGY",
    "+10 TO VITALITY",
    "5% MANA STOLEN PER HIT",
    "10% LIFE STOLEN PER HIT",
    "3% MANA STOLEN PER HIT",
    "7% MANA STOLEN PER HIT",
    "+100% DAMAGE TO UNDEAD",
    "+200% DAMAGE TO DEMONS",
    "+4 TO LIGHT RADIUS",
    "+7 TO LIGHT RADIUS",
    "REGENERATE MANA 20%",
    "LEVEL 3",
    "LEVEL 5",
    "LEVEL 18",
    "LEVEL 10",
    "5% CHANCE TO CAST LEVEL 18 CHAIN LIGHTNING ON ATTACK",
    "10% CHANCE TO CAST LEVEL 10 ICE BLAST ON STRIKING",
    "33% CHANCE TO CAST LEVEL 3 AMPLIFY DAMAGE ON STRIKING",
    "COLD RESIST +15%",
    "LIGHTNING RESIST +40%",
    "FIRE RESIST +25%",
    "REQUIRED LEVEL: 29",
    "REQUIRED LEVEL: 39",
    "REQUIRED LEVEL: 48",
    "REQUIRED LEVEL: 68",
    "SOCKETED (4)",
    "SOCKETED (5)",
    "SKULL SPLITTER",
    "+1 TO ALL SKILLS",
    "+2 TO AMAZON SKILL LEVELS",
    "+1 TO SORCERESS SKILL LEVELS",
]
numeric_words.update(game_specific)

print(f"[4.5] 数字模板: {len(numeric_words)} 个")

# ============================================================
# 5. 合并去重
# ============================================================
all_words = set()

# 物品名直接加（转大写，因为游戏实际显示全大写）
for name in item_names:
    all_words.add(name.upper())

# 属性描述直接加
for desc in property_descs:
    all_words.add(desc.upper())

# UI文本
for t in ui_templates:
    all_words.add(t.upper())

# 数字模板
all_words.update(numeric_words)

print(f"[5] 去重后总计: {len(all_words)} 个唯一词组")

# ============================================================
# 6. 标准化 & 过滤：转大写，只保留游戏实际显示的字符
# ============================================================
ALLOWED_CHARS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ :+()-%'.")
valid_words = []
invalid_words = []
seen = set()
for w in all_words:
    w_upper = w.upper()
    # 跳过重复
    if w_upper in seen:
        continue
    # 检查所有字符是否在允许集合中
    if all(c in ALLOWED_CHARS or c.isspace() for c in w_upper):
        seen.add(w_upper)
        valid_words.append(w_upper)
    else:
        invalid_words.append(w_upper)

print(f"[6] 字符过滤后: {len(valid_words)} 个有效词 (剔除 {len(invalid_words)} 个含非法字符的词)")

# ============================================================
# 7. 输出
# ============================================================

# 输出 JSON
output_json = os.path.join(PROJECT_ROOT, 'vocabulary.json')
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump({
        'items': sorted(list(item_names)),
        'properties': sorted(list(property_descs)),
        'ui_texts': sorted(list(ui_texts_set)),
        'numeric_patterns': sorted([w for w in valid_words if any(c.isdigit() for c in w)]),
        'all_words': sorted(valid_words),
    }, f, ensure_ascii=False, indent=2)
print(f"\n[输出] {output_json}")

# 输出纯文本词表（每行一个词，方便 stage2 直接读取）
output_txt = os.path.join(PROJECT_ROOT, 'vocabulary.txt')
with open(output_txt, 'w', encoding='utf-8') as f:
    for w in sorted(valid_words):
        f.write(w + '\n')
print(f"[输出] {output_txt}")

# 输出统计
print(f"\n{'='*50}")
print(f"词汇表构建完成!")
print(f"  物品名:      {len(item_names)}")
print(f"  属性描述:    {len(property_descs)}")
print(f"  UI文本:      {len(ui_texts_set)}")
print(f"  总有效词:    {len(valid_words)}")
print(f"  无效词剔除:  {len(invalid_words)}")
print(f"{'='*50}")
