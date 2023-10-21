use bevy::{
    prelude::*,
    window::{PresentMode, PrimaryWindow},
};

mod bhtree;
mod camera;
mod spawn;

use bhtree::{BHTree, Quadrant};

const WINDOW_SIZE: f32 = 1200.;
const OUT_OF_BOUNDS: f32 = 2.0; // track particles this far out of bounds

// http://arborjs.org/docs/barnes-hut
const THETA_THRESHOLD: f32 = 1.0;

const GRAVITY: f32 = 1e-6; // m^3 / (kg s^2)
const NEGATIVE_MASS: f32 = 0.; // apply expansion factor
const DSCALE: f32 = 1.0; // distance scaling w.r.t. pixels. average distance between stars in milky way is 5 light years, or 4.73e16 meters
const SIM_STEP: f32 = 0.1; // time of each sim step, in seconds. 1e12 seconds is 31.7k years. takes 230million years for sun to get around milky way.

const FPS: f64 = 120.0;
const TIME_STEP: f64 = 1.0 / FPS; // how often bevy will attempt to run the sim, in seconds

const NUM_PARTICLES: u32 = 5000;
const PARTICLE_MASS_LOWER: f32 = 1.0;
const PARTICLE_MASS_UPPER: f32 = 10.;

const VEL_VARIATION: f32 = 0.01;
const GALAXY_WIDTH_SCALE: f32 = 0.3;

const SPAWN_BLACKHOLES: bool = true;
const BLACK_HOLE_REL_MASS: f32 = 1e4;

// Minimum radius to guard against gravity singularity
const MIN_R: f32 = 0.5 * DSCALE;
const MIN_R2: f32 = MIN_R * MIN_R;

// Min grid size to protect against floating point division errors
const MIN_QUADRANT_LENGTH: f32 = 5e-3;

#[derive(Component)]
struct Pose {
    r: Vec2,
    v: Vec2,
    prev_accel: Option<Vec2>,
}

#[derive(Component, Deref)]
struct Mass(f32);

#[derive(Debug, Clone, Default, Copy, Eq, PartialEq, Hash, States)]
enum AppState {
    #[default]
    Paused,
    Running,
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::rgb(0., 0., 0.)))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "gravity".to_string(),
                resolution: (WINDOW_SIZE, WINDOW_SIZE).into(),
                present_mode: PresentMode::Immediate,
                ..default()
            }),
            ..default()
        }))
        .add_state::<AppState>()
        .add_systems(Startup, spawn::setup)
        .add_systems(
            Update,
            (detect_pause, camera::movement, bevy::window::close_on_esc),
        )
        .add_systems(Update, update.run_if(in_state(AppState::Running)))
        .run();
}

fn detect_pause(
    current_state: Res<State<AppState>>,
    mut new_state: ResMut<NextState<AppState>>,
    mut keys: ResMut<Input<KeyCode>>,
) {
    if keys.just_pressed(KeyCode::Space) {
        new_state.set(match current_state.get() {
            AppState::Running => AppState::Paused,
            AppState::Paused => AppState::Running,
        });
        keys.reset(KeyCode::Space);
    }
}

fn update(
    windows: Query<&Window, With<PrimaryWindow>>,
    par_commands: ParallelCommands,
    mut particle_query: Query<(Entity, &mut Transform, &mut Pose, &Mass)>,
) {
    let bounds = OUT_OF_BOUNDS * windows.get_single().unwrap().width();

    // each cycle, we build the quad-tree out of the particles
    let mut tree = BHTree::new(Quadrant::new(bounds * DSCALE));
    for (particle, _, pose, mass) in &particle_query {
        tree.insert(particle, Body::new(mass, pose.r))
    }

    // update the position of each particle
    particle_query
        .par_iter_mut()
        .for_each_mut(|(particle, mut transform, mut pose, mass)| {
            let Mass(mass_f) = *mass;
            // TODO avoid recreating Body in each loop
            let mut accel = tree.get_force(&particle, &Body::new(mass, pose.r)) / mass_f;
            let t = SIM_STEP;

            // apply "external mass"
            if NEGATIVE_MASS > 0. {
                let rmag = pose.r.length();
                if rmag > 0. {
                    accel += GRAVITY * NEGATIVE_MASS * pose.r.normalize_or_zero() / rmag;
                }
            }

            // https://en.wikipedia.org/wiki/Leapfrog_integration
            let dv = accel * t;
            if let Some(prev) = pose.prev_accel {
                pose.v += 0.5 * (prev * SIM_STEP + dv);
            }
            pose.r = pose.r + (pose.v + 0.5 * dv) * t;
            pose.prev_accel = Some(accel);

            // set actual rendering position
            transform.translation.x = pose.r.x / DSCALE;
            transform.translation.y = pose.r.y / DSCALE;

            // despawn particles that go out of bounds (The other option is to crash! :P or simulate far outside of bounds so they can come back)
            if (transform.translation.x.abs() >= bounds / 2.0 - 1.0)
                || (transform.translation.y.abs() >= bounds / 2.0 - 1.0)
            {
                par_commands.command_scope(|mut commands| {
                    commands.entity(particle).despawn();
                });
            }
        });
}

#[derive(Default, Clone, Copy)]
pub struct Body {
    mass: f32,
    pos: Vec2,
}

impl Body {
    fn new(mass: &Mass, pos: Vec2) -> Self {
        let Mass(mass) = *mass;
        Body { mass, pos }
    }

    fn add(&mut self, b: &Body) {
        // get total mass, but don't set overall mass until after we update COM
        let total_mass = self.mass + b.mass;
        // update center of mass
        self.pos.x = (self.pos.x * self.mass + b.pos.x * b.mass) / total_mass;
        self.pos.y = (self.pos.y * self.mass + b.pos.y * b.mass) / total_mass;
        self.mass = total_mass;
    }

    /// compute force on self from Body
    fn force(&self, b: &Body) -> Vec2 {
        let dist2 = b.pos.distance_squared(self.pos);
        // protect against tiny floating point values, otherwise we get insane acceleration without collision detection
        // Barnes. "Gravitational softening as a smoothing operation"
        // https://home.ifa.hawaii.edu/users/barnes/research/smoothing/soft.pdf
        // F_vec = -G * M1 * M2 * (r2-r1) / (|r2-r1|^2 + eps^2)^3/2
        -GRAVITY * self.mass * b.mass * (self.pos - b.pos) / (dist2 + MIN_R2).powf(3. / 2.)
    }
}
