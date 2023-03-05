use bevy::{
    prelude::*,
};
use crate::Body;
use crate::MIN_QUADRANT_LENGTH;
use crate::THETA_THRESHOLD;

enum Corner {NW, NE, SW, SE}

#[derive(Default, Debug)]
pub struct Quadrant {
    center: Vec2,
    len: f32
}

impl Quadrant {
    pub fn new(length: f32) -> Self {
        Quadrant {
            len: length,
            center: Vec2::new(0., 0.)
        }
    }
    /// return true if this Quadrant contains (x,y)
    fn contains(&self, x: f32, y: f32) -> bool {
        let hl = self.len / 2.0;
        (x >= self.center.x - hl) && (x < self.center.x + hl) && (y >= self.center.y - hl) && (y < self.center.y + hl)
    }
    
    fn subquad(&self, corner: Corner) -> Self {
        let hl = self.len / 2.0;
        let ql = hl / 2.0;
        match corner {
            Corner::NW => Quadrant {center: Vec2::new(self.center.x - ql, self.center.y + ql), len: hl},
            Corner::NE => Quadrant {center: Vec2::new(self.center.x + ql, self.center.y + ql), len: hl},
            Corner::SW => Quadrant {center: Vec2::new(self.center.x - ql, self.center.y - ql), len: hl},
            Corner::SE => Quadrant {center: Vec2::new(self.center.x + ql, self.center.y - ql), len: hl}
        }
    }
}


enum NodeItem {
    Internal(SubQuadrants),
    Leaf(Entity)
}

struct Node {
    body: Body,
    item: NodeItem
}

impl Node {
    fn new(body: Body, item: NodeItem) -> Self {
        Node {body, item}
    }
}

#[derive(Default)]
pub struct BHTree
{
    quad: Quadrant,
    node: Option<Node>
}

impl BHTree
{
    pub fn new(quad: Quadrant) -> Self {
        BHTree {quad, ..default()}
    }

    pub fn insert(&mut self, particle: Entity, body: Body) {
        if let Some(current_node) = &mut self.node {
            match &mut current_node.item {
                NodeItem::Internal(subquad) => {
                    current_node.body.add(&body);
                    subquad.insert_to_quadrant(particle, body);
                },
                NodeItem::Leaf(node_particle) => {
                    if self.quad.len > MIN_QUADRANT_LENGTH {
                        // we only have 1 particle per region, so we split and generate subtrees
                        let mut subquad = SubQuadrants::new(&self.quad);
                        subquad.insert_to_quadrant(node_particle.clone(), current_node.body);
                        subquad.insert_to_quadrant(particle, body);
                        current_node.item = NodeItem::Internal(subquad);
                    }
                    // implied else: if we've already got too small of a grid, we still add the mass for a cheap estimate
                    current_node.body.add(&body);
                }
            }
        } else {
            // current node has no body, add it here as particle/external
            self.node = Some(Node::new(body, NodeItem::Leaf(particle)));
        }
    }

    pub fn get_force(&self, p: &Entity, b: &Body) -> Vec2 {
        if let Some(current_node) = &self.node {
            match &current_node.item {
                NodeItem::Internal(subquad) => {
                    let dist = current_node.body.pos.distance(b.pos);
                    if self.quad.len / dist < THETA_THRESHOLD {
                        // treat node as a single body
                        b.force(&current_node.body)
                    } else {
                        // traverse the tree, returning the total force
                        subquad.get_force(p, b)
                    }
                },
                NodeItem::Leaf(node_particle) => {
                    if node_particle.index() != p.index() {
                        b.force(&current_node.body)
                    } else {
                        // index was the same, this is the same particle
                        Vec2::new(0., 0.)
                    }
                }
            }
        } else {
            // there's no body at self, so there's no force
            Vec2::new(0., 0.)
        }
    }
}


struct SubQuadrants {
    nw: Box<BHTree>,
    ne: Box<BHTree>,
    sw: Box<BHTree>,
    se: Box<BHTree>
}

impl SubQuadrants {
    fn new(q: &Quadrant) -> Self {
        SubQuadrants {
            nw: Box::new(BHTree::new(q.subquad(Corner::NW))),
            ne: Box::new(BHTree::new(q.subquad(Corner::NE))),
            sw: Box::new(BHTree::new(q.subquad(Corner::SW))),
            se: Box::new(BHTree::new(q.subquad(Corner::SE))),
        }
    }

    fn get_force(&self, p: &Entity, b: &Body) -> Vec2 {
        self.nw.get_force(p, b) + self.ne.get_force(p, b) + self.sw.get_force(p, b) + self.se.get_force(p, b)
    }

    fn insert_to_quadrant(&mut self, p: Entity, b: Body) {
        // this is an internal node, we must have a subtree
        match b {
            b if self.nw.quad.contains(b.pos.x, b.pos.y) => self.nw.insert(p, b),
            b if self.ne.quad.contains(b.pos.x, b.pos.y) => self.ne.insert(p, b),
            b if self.sw.quad.contains(b.pos.x, b.pos.y) => self.sw.insert(p, b),
            b if self.se.quad.contains(b.pos.x, b.pos.y) => self.se.insert(p, b),
            b => panic!("position {}, {} was not in any quadrant?\n {:#?}, {:#?}, {:#?}, {:#?}", b.pos.x, b.pos.y, self.nw.quad, self.ne.quad, self.sw.quad, self.se.quad)
        }
    }
}