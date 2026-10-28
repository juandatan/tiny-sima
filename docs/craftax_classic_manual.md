# Craftax-Classic: player's manual

Craftax-Classic is a top-down survival and crafting game on a single, randomly
generated world. You see a small window of the map around your character, plus
your vitals and your inventory. There is no fixed goal: the game rewards you
the first time you unlock each achievement, and the episode ends when you die
or run out of time.

This manual describes what things are, what they are for and where they must
be used. It deliberately says nothing about how many of anything a recipe
takes: you will have to find that out by playing.

## The world

The world is a grid of tiles. Each tile holds one block:

- **Grass**: open ground you can walk on. Interacting with grass sometimes
  yields a sapling.
- **Sand**: open ground, usually found beside water.
- **Water**: lakes and rivers. You cannot walk on water, but you can drink from
  it.
- **Tree**: a source of wood. Trees block movement. In this version, a tree you
  chop down is gone for good, so wood is limited.
- **Stone**: rock in mountains and caves. It blocks movement until mined, and
  mining it leaves a walkable path.
- **Path**: the floor left behind where stone was mined, and the floor of
  tunnels.
- **Coal**, **iron** and **diamond**: ores embedded in stone, mostly deep in the
  mountains and caves. Diamond is the rarest.
- **Lava**: found in caves. Walking into lava kills you.
- **Crafting table** and **furnace**: stations you build yourself. No world
  starts with one.
- **Plant**: a sapling you planted. It grows over time into a ripe plant.

## Your vitals

You have four vitals, each shown as a bar:

- **Health** is lost to attacks and when any other vital is empty. It slowly
  comes back by itself while food, drink and energy are all above empty. When
  health runs out you die.
- **Food** drains over time. Eat a cow or a ripe plant to restore it.
- **Drink** drains over time. Face water and interact with it to drink; each
  drink restores a little.
- **Energy** drains over time while you are awake. Sleep to restore it. While
  asleep you are vulnerable, and you wake when energy is restored or when you
  are attacked.

Every vital has a maximum, and restoring a full vital does nothing.

## Day and night

The world cycles between day and night. At night the screen darkens, your view
is harder to read, and zombies appear far more often on grass. Sleeping
through the night in a sheltered spot is a common strategy.

## Creatures

- **Cows** wander the grassland. Hit a cow until it dies to eat it.
- **Zombies** live on grass, especially at night. They walk towards you and
  attack at close range. You can fight back.
- **Skeletons** live in the dark tunnels of caves. They keep their distance and
  shoot arrows. You can fight back.

Weapons make you hit harder; better swords kill faster.

## Actions

- **Move** left, right, up or down. You also turn to face that direction; if
  the tile is blocked you only turn.
- **Noop**: do nothing for one step.
- **Do** (interact) with the tile you are facing: chop a tree, mine stone or
  ore (with the right pickaxe), drink water, collect a sapling from grass, eat
  a ripe plant, or attack a creature.
- **Sleep**.
- **Place stone** on the tile you face, for example to bridge water, block
  lava or wall yourself in. It uses stone from your inventory.
- **Place table**: build a crafting table on the tile you face. It uses wood.
- **Place furnace**: build a furnace on the tile you face. It uses stone.
  Build it near your crafting table: iron tools need both stations nearby.
- **Place plant**: plant a sapling on a grass tile you face.
- **Make wood pickaxe**, **make stone pickaxe**, **make iron pickaxe**.
- **Make wood sword**, **make stone sword**, **make iron sword**.

The place actions need a free tile in front of you (grass, sand or path; stone
can also go into water or lava). The make actions need the right station
nearby: "nearby" means within a few tiles of you, not necessarily adjacent.
An action whose requirements are not met simply does nothing.

## Materials and what they need

| Material | Where it comes from | Tool needed to collect it |
| --- | --- | --- |
| Wood | trees | none (bare hands) |
| Sapling | grass, by chance | none |
| Stone | stone blocks | wood pickaxe |
| Coal | coal ore | wood pickaxe |
| Iron | iron ore | stone pickaxe |
| Diamond | diamond ore | iron pickaxe |
| Drink | water | none |

Holding a tool is enough: tools are never used up by mining. Every item count
in your inventory has a cap.

## Stations and recipes

Crafting consumes the listed materials from your inventory. Stations are not
consumed: once built, a table or furnace can be used again and again as long
as you stay near it.

| Item | Made from | Where |
| --- | --- | --- |
| Crafting table | wood | placed on a free tile in front of you |
| Furnace | stone | placed on a free tile in front of you |
| Wood pickaxe | wood | near a crafting table |
| Wood sword | wood | near a crafting table |
| Stone pickaxe | wood and stone | near a crafting table |
| Stone sword | wood and stone | near a crafting table |
| Iron pickaxe | wood, stone, coal and iron | near a crafting table and a furnace |
| Iron sword | wood, stone, coal and iron | near a crafting table and a furnace |

Recipes differ in how much of each material they take, and this manual does
not list the amounts. Plan your wood carefully, because trees do not grow
back.

## Achievements

Each achievement is unlocked once, the first time you do the thing it names:

- Gathering: collect wood, collect sapling, collect drink, collect stone,
  collect coal, collect iron, collect diamond.
- Building: place table, place furnace, place stone, place plant.
- Crafting: make wood pickaxe, make wood sword, make stone pickaxe, make stone
  sword, make iron pickaxe, make iron sword.
- Survival: eat cow, eat plant, wake up (sleep and wake again).
- Combat: defeat zombie, defeat skeleton.

The tech tree runs wood, then a table, then wooden tools, then stone, then
stone tools and a furnace, then coal and iron, then iron tools, and finally
diamond. Staying fed, watered and rested along the way keeps you alive long
enough to climb it.
