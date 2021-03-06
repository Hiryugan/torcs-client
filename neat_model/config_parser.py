import errno
import yaml
from lxml import etree
import os


class Parser:
    def __init__(self, parse_file):
        self.parse_file = parse_file
        with open(self.parse_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.CLoader)
            self.neat_config_file = config['neat_config_file']
            self.output_model = config['output_model']
            self.port = config['port']
            self.save_path = config['save_path']
            self.server_config_file = config['server_config_file']
            self.tracks = config['tracks']
            self.fitness_function_file = config['fitness_function_file']
            self.merge_accel_brake = config['merge_accel_brake']
            self.avoid_go_out = config['avoid_go_out']
        f.close()

class Configurator:
    def __init__(self, file):
        self.parser = Parser(file)

    def configure_server(self):
        file = '/home/hiryugan/Documents/torcs-server/example_torcs_config.xml'
        xml_tree = etree.parse(file)
        tracks = xml_tree.xpath("//section[@name='Tracks']")[0]
        attnum_tracks = tracks.xpath("attnum")[0]
        attnum_tracks.attrib['val'] = str(len(self.parser.tracks))
        for section in tracks.xpath("section"):
            section.getparent().remove(section)

        for idx, track_data in enumerate(self.parser.tracks):
            track_name = track_data['name']
            track_category = track_data['category']
            section = etree.SubElement(tracks, 'section', {'name': str(idx + 1)})
            track_name_tag = etree.SubElement(section, 'attstr', {'name': 'name', 'val': track_name})
            track_category_tag = etree.SubElement(section, 'attstr', {'name': 'category', 'val': track_category})

        drivers_section = xml_tree.xpath("//section[@name='Drivers']/section")[0]
        client_port = drivers_section.xpath("attnum")[0]
        client_port.attrib['val'] = str(self.parser.port)

        server_config_file = self.parser.server_config_file
        dirname = os.path.dirname(server_config_file)
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                   raise
        xml_tree.write(server_config_file, pretty_print=True)

    def configure_client(self):
        dirname = os.path.dirname(self.parser.save_path)
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

